"""
sigmoid.py —— 编译并基准测试 sigmoid.cu 中的 6 个 CUDA kernel

流程：
  1. 用 torch.utils.cpp_extension.load 现场编译 sigmoid.cu；
  2. 对多组 (S, K) 形状生成随机张量（FP32 与 FP16）；
  3. run_benchmark() 先 warmup，再重复执行取平均耗时；
  4. 与 PyTorch 官方 torch.sigmoid 对比正确性（打印前 2 个结果）和性能。

运行前建议指定目标 GPU 架构以缩短编译时间，例如：
  export TORCH_CUDA_ARCH_LIST=Ada
  python3 sigmoid.py
"""

import time
from functools import partial
from typing import Optional

import torch
from torch.utils.cpp_extension import load
import torch.utils.cpp_extension as ext

# sigmoid 不需要反向传播，关闭 autograd 可以省去图构建开销
torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="sigmoid_lib",
    sources=["sigmoid.cu"],
    # 常用 CUDA 编译选项：
    #   -O3                            最高优化等级
    #   -U__CUDA_NO_HALF_OPERATORS__ 等 恢复 half / bfloat16 的运算符与转换，
    #                                   便于在 C++ 侧直接写 half 运算
    #   --expt-relaxed-constexpr        允许 constexpr 中调用某些设备函数
    #   --expt-extended-lambda          允许 __device__ 扩展 lambda
    #   --use_fast_math                 使用快速数学库（牺牲少量精度换取速度；
    #                                   sigmoid 的开销主要在 exp 上，收益明显）
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

# 打印 PyTorch CUDA 扩展的实际构建目录（sigmoid_lib.so 所在位置）
print(ext._get_build_directory("sigmoid_lib", False))

# 打印当前 GPU 设备名
print(torch.cuda.get_device_name())


# 基准测试封装：
# - 传入 out 时走“原地写 y”的 kernel 接口（本模块 kernel 签名是 (x, y)）；
# - 不传 out 时函数返回新张量（如 torch.sigmoid 默认接口）；
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
    # warmup：先跑若干次，让 kernel 加载、GPU 频率和缓存达到稳定状态
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
    total_time = (end - start) * 1000  # 单位换算为毫秒
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    # GPU 张量 → 拉平 → 断开可能的梯度追踪（本脚本已关 autograd，属保险写法）
    # → 拷回 CPU → 转 numpy → 转 Python 列表，再取前 2 个元素
    # 只打印前 2 个元素用于人工核对正确性（float / half 数值都可读）
    out_val = out.flatten().detach().cpu().numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]
    print(f"{out_info:>18}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time


# 测试规模：模拟类似矩阵 (S, K) 的形状，内部按 N = S * K 展平计算
Ss = [1024, 2048, 4096]
Ks = [1024, 2048, 4096]
SKs = [(S, K) for S in Ss for K in Ks]

for S, K in SKs:
    print("-" * 85)
    print(" " * 40 + f"S={S}, K={K}")
    # FP32 测试：连续内存布局是向量化访存的前提，所以显式 .contiguous()
    x = torch.randn((S, K)).cuda().float().contiguous()
    y = torch.zeros_like(x).cuda().float().contiguous()
    # 依次测：标量 / float4 向量化 / PyTorch 官方实现
    run_benchmark(lib.sigmoid_f32, x, "f32", y)
    run_benchmark(lib.sigmoid_f32x4, x, "f32x4", y)
    run_benchmark(partial(torch.sigmoid, out=y), x, "f32_th")

    print("-" * 85)
    # FP32 尾数 23 位，最小刻度 2^-23 ≈ 0.00000012（约 7 位有效数字）；
    # FP16 尾数只有 10 位，最小刻度 2^-10 = 1/1024 ≈ 0.00098
    # （约 3~4 位有效数字，保守按约 3 位），FP32 转 half 会就近取整造成量化误差，
    # 属于预期精度损失；sigmoid 结果落在 (0, 1)，差异体现在小数点后第 4 位左右。
    # FP16 测试：从同一组随机数转成 half，便于对比精度损失
    x_f16 = x.half().contiguous()
    y_f16 = y.half().contiguous()
    # 依次测：标量 / half2 / 8 元素 / 128 位打包 / PyTorch 官方实现
    run_benchmark(lib.sigmoid_f16, x_f16, "f16", y_f16)
    run_benchmark(lib.sigmoid_f16x2, x_f16, "f16x2", y_f16)
    run_benchmark(lib.sigmoid_f16x8, x_f16, "f16x8", y_f16)
    run_benchmark(lib.sigmoid_f16x8_pack, x_f16, "f16x8pack", y_f16)
    run_benchmark(partial(torch.sigmoid, out=y_f16), x_f16, "f16_th")
    print("-" * 85)
