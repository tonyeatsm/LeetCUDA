"""
gelu.py —— 编译并基准测试 gelu.cu 中的 6 个 CUDA kernel

流程：
  1. 用 torch.utils.cpp_extension.load 现场编译 gelu.cu；
  2. 对多组 (S, K) 形状生成随机张量（FP32 与 FP16）；
  3. run_benchmark() 先 warmup，再重复执行取平均耗时；
  4. 与 PyTorch 官方 tanh 近似 GELU（torch.nn.GELU("tanh")）对比正确性和性能。

注意：GELU 的 tanh 近似在 FP16 下用 hexp 拼实现，存在相减抵消的精度损失；
当输入 x ≳ 4.03 时 hexp 还会溢出成 inf、算出 NaN（详见
docs/kernels/gelu/README.md 与 kernels/gelu/README.md 的说明）。

运行前建议指定目标 GPU 架构以缩短编译时间，例如：
  export TORCH_CUDA_ARCH_LIST=Ada
  python3 gelu.py
"""

import time
from functools import partial
from typing import Optional

import torch.nn
import torch.utils
from torch.utils.cpp_extension import load
import torch.utils.cpp_extension as ext

# gelu 不需要反向传播，关闭 autograd 可以省去图构建开销
torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="gelu_lib",
    sources=["gelu.cu"],
    # 常用 CUDA 编译选项：
    #   -O3                            最高优化等级
    #   -U__CUDA_NO_HALF_OPERATORS__ 等 恢复 half / bfloat16 的运算符与转换，
    #                                   便于在 C++ 侧直接写 half 运算
    #   --expt-relaxed-constexpr        允许 constexpr 中调用某些设备函数
    #   --expt-extended-lambda          允许 __device__ 扩展 lambda
    #   --use_fast_math                 使用快速数学库（牺牲少量精度换取速度；
    #                                   GELU 的开销主要在 tanh 上，收益明显；
    #                                   但它也让 half 版的溢出行为更"干脆"，
    #                                   x ≳ 4.03 时直接得到 NaN）
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)

# 打印 PyTorch CUDA 扩展的实际构建目录（gelu_lib.so 所在位置）
print(ext._get_build_directory("gelu_lib", False))

# 打印当前 GPU 设备名
print(torch.cuda.get_device_name())


# 基准测试封装：
# - 传入 out 时走"原地写 y"的 kernel 接口（本模块 kernel 签名是 (x, y)）；
# - 不传 out 时函数返回新张量（torch.nn.GELU 是模块实例，没有 out 参数）；
# - 先 warmup 让 GPU 时钟与缓存达到稳定状态，再正式计时取平均。
def run_benchmark(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
):
    # 清空输出张量，避免上一次测试的残留数据影响结果
    if out is not None:
        out.fill_(0)
    # warmup：先跑若干次，让 kernel 加载、GPU 频率和缓存达到稳定状态。
    # 对照实现（nn.Module）没有 out 参数，因此这里区分了两条调用路径。
    if out is not None:
        for i in range(warmup):
            perf_func(x, out)
    else:
        for i in range(warmup):
            _ = perf_func(x)
    # 同步：确保前面所有 GPU 任务执行完，CPU 计时点才是准确的
    torch.cuda.synchronize()

    start = time.time()
    # 正式计时：连续执行 iters 次，最后取平均单次耗时
    if out is not None:
        for i in range(iters):
            perf_func(x, out)
    else:
        for i in range(iters):
            out = perf_func(x)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    # GPU 张量 → 拉平 → 断开可能的梯度追踪（本脚本已关 autograd，属保险写法）
    # → 拷回 CPU → 转 numpy → 转 Python 列表，再取前 2 个元素
    out_val = out.flatten().detach().cpu().numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]
    # 只打印前 2 个元素用于人工核对正确性
    # （注意：本模块没有做 f"{v:<12}" 左对齐补齐，
    #   所以 README 里的输出是干净的 [-0.13358943, -0.06881647] 形式）
    print(f"{out_info:>18}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time


# 测试规模：模拟类似矩阵 (S, K) 的形状，内部按 N = S * K 展平计算
Ss = [1024, 2048, 4096]
Ks = [1024, 2048, 4096]
SKs = [(S, K) for S in Ss for K in Ks]
# PyTorch 的对照片：torch.nn.GELU("tanh") 是一个 nn.Module **实例**，
# 必须显式写 "tanh" 才能和本模块的 tanh 近似对齐
# （默认是 approximate='none'，即 erff 精确式，两者会有可见差异）。
# 把模块实例赋给 torch.gelu 这个名字是脚本里的临时写法，
# 下面再用 partial(...) 包一层，让调用形式与其它模块保持一致。
# 该模块实例没有 out 参数，所以对照测试走 run_benchmark 的"不传 out"分支。
torch.gelu = torch.nn.GELU("tanh")
for S, K in SKs:
    print("-" * 85)
    print(" " * 40 + f"S={S}, K={K}")
    # FP32 测试：连续内存布局是向量化访存的前提，所以显式 .contiguous()
    x = torch.randn((S, K)).cuda().float().contiguous()
    y = torch.zeros_like(x).cuda().float().contiguous()
    # 依次测：标量 / float4 向量化 / 对照实现
    run_benchmark(lib.gelu_f32, x, "f32", y)
    run_benchmark(lib.gelu_f32x4, x, "f32x4", y)
    run_benchmark(partial(torch.gelu), x, "f32_th")
    print("-" * 85)
    # FP16 测试：从同一组随机数转成 half。
    # 量化误差 + hexp 拼 tanh 的相减抵消都会体现在这一组里；
    # 另外 torch.randn 在大尺寸下会出现个别 |x| > 4.03 的元素，
    # 它们会让本模块的 FP16 结果变成 NaN（抽查前 2 个元素看不出来）。
    x_f16 = x.half().contiguous()
    y_f16 = y.half().contiguous()
    # 依次测：标量 / half2 / 8 元素 / 128 位打包 / 对照实现
    run_benchmark(lib.gelu_f16, x_f16, "f16", y_f16)
    run_benchmark(lib.gelu_f16x2, x_f16, "f16x2", y_f16)
    run_benchmark(lib.gelu_f16x8, x_f16, "f16x8", y_f16)
    run_benchmark(lib.gelu_f16x8_pack, x_f16, "f16x8pack", y_f16)
    run_benchmark(partial(torch.gelu), x_f16, "f16_th")
    print("-" * 85)
