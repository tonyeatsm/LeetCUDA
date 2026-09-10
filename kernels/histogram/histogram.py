"""
histogram.py —— 编译并测试 histogram.cu 中的 2 个 CUDA kernel

流程：
  1. 用 torch.utils.cpp_extension.load 现场编译 histogram.cu；
  2. 打印 PyTorch CUDA 扩展的实际构建目录（hist_lib.so 所在位置）；
  3. 构造 a = [0, 1, ..., 9] * 1000（长度 10000，元素均为非负整数）；
  4. 分别调用 histogram_i32（标量版）与 histogram_i32x4（int4 向量化版）；
  5. 打印每个桶的计数值，人工核对每个值是否都出现 1000 次。

运行前建议指定目标 GPU 架构以缩短编译时间，例如：
  export TORCH_CUDA_ARCH_LIST=Blackwell   # 或 Ada
  python3 histogram.py
"""

import torch
from torch.utils.cpp_extension import load

# 用于查询 PyTorch CUDA 扩展的实际构建目录
import torch.utils.cpp_extension as ext

# 直方图统计不需要反向传播，关闭 autograd 可以省去图构建开销
torch.set_grad_enabled(False)

# 现场编译 histogram.cu，得到一个 Python 可调用模块 lib
# Load the CUDA kernel as a python module
lib = load(
    name="hist_lib",
    sources=["histogram.cu"],
    # 常用 CUDA 编译选项（与 elementwise.py 保持一致）：
    #   -O3                            最高优化等级
    #   -U__CUDA_NO_HALF_OPERATORS__ 等 恢复 half / bfloat16 的运算符与转换，
    #                                   便于在 C++ 侧直接使用 half 相关类型
    #   --expt-relaxed-constexpr        允许 constexpr 中调用某些设备函数
    #   --expt-extended-lambda          允许 __device__ 扩展 lambda
    #   --use_fast_math                 使用快速数学库（牺牲少量精度换取速度）
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

# 打印 PyTorch CUDA 扩展的实际构建目录（hist_lib.so 所在位置）
print(ext._get_build_directory("hist_lib", False))

# 构造测试数据：0~9 每个值各出现 1000 次，总长度 N = 10 * 1000 = 10000。
# 元素都是非负 int32；且 N 恰好是 4 的倍数，因此向量化版本不会读到越界数据。
a = torch.tensor(list(range(10)) * 1000, dtype=torch.int32).cuda()

# 标量版：host 端用 max(a) + 1 = 10 决定桶的数量，一个线程处理 1 个元素
h_i32 = lib.histogram_i32(a)
print("-" * 80)
for i in range(h_i32.shape[0]):
    print(f"h_i32   {i}: {h_i32[i]}")

# 向量化版：一个线程处理 4 个元素（int4，16 字节访存），预期结果与标量版一致
print("-" * 80)
h_i32x4 = lib.histogram_i32x4(a)
for i in range(h_i32x4.shape[0]):
    print(f"h_i32x4 {i}: {h_i32x4[i]}")
print("-" * 80)
