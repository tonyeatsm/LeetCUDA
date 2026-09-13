// ============================================================================
// swish.cu —— CUDA Swish / SiLU（自门控激活函数）教学示例
//
// 本文件实现 y[i] = swish(x[i])，数组长度为 N：
//
//   swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// 命名说明：swish 与 silu 是同一个函数的两个名字。Google 在 2017 年的论文里
// 把它叫 Swish（x * sigmoid(beta * x)，beta = 1 时即本文件的形式），
// PyTorch 沿用了 SiLU 的叫法（torch.nn.SiLU / torch.nn.functional.silu）。
// 本文件沿用仓库里已有的 swish 命名。
//
// 所有 kernel 都把输入输出当作“一维数组”看待，二维 (S, K) 矩阵在启动端
// 展平成 N = S * K。
//
// 文件中包含 6 个计算版本：
//   1. f32         : FP32 标量版（每线程 1 个元素）
//   2. f32x4       : FP32 向量化版（每线程 4 个元素，16 字节访存）
//   3. f16         : FP16 标量版（half，每线程 1 个元素）
//   4. f16x2       : FP16 每线程 2 个元素（half2，4 字节访存）
//   5. f16x8       : FP16 每线程 8 个元素（4 次 half2 操作，unpack 写法）
//   6. f16x8_pack  : FP16 每线程 8 个元素（按 128 位打包成 1 次访存）
//
// 与 elementwise / sigmoid / relu 三个模块的关系：
// 本文件的 6 个版本与 kernels/elementwise/elementwise.cu、
// kernels/sigmoid/sigmoid.cu、kernels/relu/relu.cu 的 6 个版本一一对应，
// 区别只是把 “c[i] = a[i] + b[i]” / “y[i] = sigmoid(x[i])” /
// “y[i] = max(0, x[i])” 换成 “y[i] = swish(x[i])”。
// 线程编号、向量化访存、FP16 打包等通用知识不再重复，下面重点说明
// Swish 特有的三点：
//   - 它把 sigmoid 用在了自己身上（x 乘上自己的门控），可以看作 sigmoid
//     模块的“进阶版”：同样是 exp，再多一步加法、除法和乘法；
//   - kernel 里没有真的先算 sigmoid 再乘，而是用等价的
//     x / (1 + exp(-x)) 一次算完，省掉一次中间变量；
//   - 不需要 clamp：exp(-x) 溢出成 inf 时结果恰好是数学极限 0，
//     不会像 gelu 那样出现 inf/inf = NaN（详见下面的宏定义段）。
//
// 学习重点：
// - 全局线程编号 = blockIdx.x * blockDim.x + threadIdx.x
// - 越界保护：if (idx < N)
// - 溢出未必是坏事：分母上的 exp 溢出恰好对应 sigmoid(-∞) = 0
// - 向量化 / 更宽访存 / FP16 都是逐元素算子的常见带宽优化手段
//
// 文件后半部分是 PyTorch 绑定：把 CUDA kernel 包装成 Python 可调用的函数，
// 供 swish.py 编译与基准测试使用。
// ============================================================================

// ============================================================================
// 【Swish 曲线速览】一张小图看懂这个激活函数
//
//   y = x / (1 + e^(-x)) = x * sigmoid(x)
//
//  3.0 ┤                          ***
//      ┤                       ***
//  2.0 ┤                    ***
//      ┤                 ***
//  1.0 ┤              ***
//      ┤           ***
//  0.0 ┤*******+
//      ┤      ***
// -0.28┤   *
//      └──────────────────────────────→ x
//      -3   -2   -1   0   1   2   3
//
// 采样值：-3→-0.1423、-2→-0.2384、-1.279→-0.2785、-1→-0.2689、
//         0→0、1→0.7311、2→1.7616、3→2.8577、4→3.9281
//
// 读图要点：
// - 正半轴近似恒等映射：x 稍大（约 3 以上）后 sigmoid(x) ≈ 1，
//   swish(x) ≈ x，与 ReLU / ELU / GELU 的正半轴一致；
// - 负半轴不是硬零：x < 0 时输出是小负数，x → -∞ 时趋近 0
//   （因为 x * 0 里 sigmoid(x) 衰减得比 x 增长更快）；最小值约 -0.2785
//   （出现在 x ≈ -1.279），所以 Swish 不是单调函数；
// - 它是“自门控”的：sigmoid(x) 扮演门控，x 为正时门开（信号通过），
//   x 为负时门关（信号被压制）——这就是 Swish 与 GELU 被称为
//   self-gated / 平滑 ReLU 家族的原因；
// - 在 x = 0 处光滑可导：swish'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))，
//   在 x = 0 处等于 0.5（注意导数不为 0，这与 ReLU 的次梯度取 0 不同）；
// - 与 GELU 的关系：两者形状很接近（先下凹再单调上升），GELU 可以看成
//   “用正态 CDF 当门控”，Swish 用的是 sigmoid 门控；Swish 的凹陷更深
//   （-0.2785 对 GELU 的 -0.1700），但计算更便宜（没有 tanh 里的三次多项式）。
//
// 与 sigmoid / relu 的对照（下文会反复引用这张表）：
//   | 对比项     | relu              | gelu (tanh 近似)        | swish / silu            |
//   | 公式       | max(0, x)         | 0.5x(1+tanh(...))       | x / (1 + exp(-x))       |
//   | 值域       | [0, +∞)，仅负端   | (-0.17, +∞)             | (-0.2785, +∞)           |
//   | 最小值位置 | x <= 0 全为 0     | x ≈ -0.752              | x ≈ -1.279              |
//   | x=0 处     | 不可导（左0右1）  | 光滑可导                | 光滑可导，导数 0.5      |
//   | 每元素计算 | 一次取最大值      | 多次乘加 + 一次 tanh    | 一次 exp + 加/除/乘     |
//   | 溢出保护   | 不需要            | 需要 clamp              | 不需要（溢出对应极限值）|
// ============================================================================

#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

// 头文件列表与 elementwise / sigmoid / relu 保持一致（其中若干头文件本模块
// 没有直接使用，保留是为了方便几个模块对照阅读）。
// 注意本文件只包含了 cuda_fp16.h，并没有包含 cuda_bf16.h，
// 因此下面的 BFLOAT2 宏（依赖 __nv_bfloat162 类型）在本文件中无法展开；
// 好在它从未被使用，所以不影响编译。
#define WARP_SIZE 32

// ----------------------------------------------------------------------------
// 访存“重解释（reinterpret）”宏
//
// GPU 一次访存指令能搬移的位宽有限。把一段连续内存 reinterpret 成更宽的
// CUDA 内置向量类型，可以让编译器生成更宽的 load/store 指令。
//
// 用法示例：FLOAT4(x[idx]) 把 &x[idx] 起的 16 字节当成一个 float4 读写。
// reinterpret_cast<T*>(&(value)) 取得该内存的 T*，[0] 取出第 1 个 T。
// 要求 value 对应地址按 16 字节（或相应类型大小）对齐，torch 分配的张量
// 通常满足要求。
//
// 说明：INT4 与 BFLOAT2 本模块没有使用，保留是为了和其它模块的宏定义段
// 保持一致；LDST128BITS 复用 float4，一次读写 128 位 = 16 字节，
// 即 8 个 half 组成的打包数据。
// ----------------------------------------------------------------------------
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// ============================================================================
// Swish 计算函数：为什么写成两个 __device__ __forceinline__ 辅助函数？
// ============================================================================
//   __device__      ：只能在设备（GPU）代码里调用，不能在 host 上调用；
//   __forceinline__ ：要求编译器强制内联，避免函数调用开销
//                     （逐元素算子每个线程都要调用一次，这点很关键）。
//
// FP32 版直接叫 swish、FP16 版叫 swish_half，与 elu 模块的
// elu / elu_half 命名习惯一致。
//
// ----------------------------------------------------------------------------
// 为什么 Swish 不需要 sigmoid / gelu 那样的 clamp（溢出保护）？
// ----------------------------------------------------------------------------
// 本模块的公式里有除法 x / (1 + exp(-x))，但两个极端都对应正确的数学极限：
//
//   | 输入              | exp(-x)            | 结果          | 说明                |
//   | x 很大的正数(88)   | exp(-88) ≈ 6e-39   | x / 1 = x     | sigmoid(88) ≈ 1     |
//   | x 很大的负数(-88)  | exp(88) 溢出为 inf | x / inf = -0  | 数学极限就是 x*0    |
//   | x = 0             | exp(0) = 1         | 0 / 2 = 0     | 恰好过原点          |
//   | x = ±inf          | 0 / inf            | ±inf / NaN    | 与 PyTorch 行为一致 |
//   | x = NaN           | NaN                | NaN           | NaN 正常传播        |
//
// 关键在第二行：exp(-x) 溢出得到的 inf 恰好对应 sigmoid(-∞) = 0，
// 而 x / inf = 0 正是极限值，所以这里的溢出是“无害”的；
// 不像 gelu 那样会算出 inf / inf = NaN。
// 这是一个很好的对照：同样的 exp，放在分母上安全，
// 放进 (exp(2t) - 1) / (exp(2t) + 1) 这种自己除自己的形式就不安全。
// ----------------------------------------------------------------------------
// FP32
// Swish x: N, y: N y=x*sigmoid(x)

// 直接写成除法：x * (1 / (1 + exp(-x))) 与 x / (1 + exp(-x)) 等价，
// 但后者只做一次除法、不需要额外的乘法与中间变量，寄存器压力更小。
// expf 是 float 版本（对应 double 版 exp）。
__device__ __forceinline__ float swish(float x) {
  return x / (1.0f + expf(-x));
}

// ============================================================================
// Kernel 1：FP32 标量版（最朴素写法，作为正确性基准）
// ============================================================================
//
// ---- 调用链：谁触发了这个 kernel？ ----
// swish.py
//   └─ lib.swish_f32(x, y)                            # Python 调用（pybind11）
//       └─ swish_f32(torch::Tensor x, y)              # C++ host 包装函数（宏生成，
//                                         #  运行在 CPU 上）
//           └─ swish_f32_kernel<<<grid, block>>>(...) # CUDA 启动语法
//               └─ cudaLaunchKernel(...)              # nvcc 生成，交给 CUDA runtime
//                   └─ GPU 硬件并行执行本函数体
//
// 关键点：
// - swish_f32 是 CPU 端包装函数，负责类型检查、计算 grid/block；
//   它在源码里由宏 TORCH_BINDING_SWISH 生成，直接搜索不到定义。
// - swish_f32_kernel 是 GPU 端 __global__ 函数，
//   只能用 <<<grid, block>>> 启动，不能像普通函数那样直接调用。
// - 启动后 GPU 的每个线程都会执行一遍本函数体，
//   靠 threadIdx / blockIdx 算出自己负责的下标 idx。
//
// ---- grid / block 知识点速查 ----
// 1. 一次 kernel 启动 <<<grid, block>>> 只创建一个 grid；
//    grid 就是“本次启动要创建的所有 block 的集合”。
// 2. 尖括号两个配置的含义（标准写法 <<<Dg, Db>>>）：
//    - grid  : 本 grid 一共几个 block，进入 kernel 后 = gridDim.x；
//    - block : 每个 block 几个线程，进入 kernel 后 = blockDim.x；
//    grid/block 只是本文件里的 dim3 局部变量名，不是 CUDA 关键词，
//    更清晰的命名是 <<<blocksPerGrid, threadsPerBlock>>>。
// 3. kernel 内部固定内置变量：
//    gridDim.x  = 总 block 数；
//    blockIdx.x = 当前 block 是第几个（0 起，在本次 grid 内唯一）；
//    blockDim.x = 每 block 线程数；
//    threadIdx.x= 线程在 block 内第几号（0 起）。
// 4. 全局线程编号 idx = blockIdx.x * blockDim.x + threadIdx.x；
//    只需知道“当前 block 前面有几个 block”，不需要知道 gridDim.x。
// 5. 例：<<<4, 256>>>、N = 1000：
//    gridDim.x=4, blockDim.x=256，总线程数 = 1024；
//    block0→idx 0~255, block1→256~511, block2→512~767, block3→768~1023；
//    idx=1000~1023 共 24 个线程被 if (idx < N) 拦截，不参与计算。
// 6. 同一个 kernel 函数可被多次启动，每次启动产生一个独立 grid；
//    准确说法是“一次启动 = 一个 grid”，而不是“一个 kernel = 一个 grid”。
//
// 约定启动方式：每个 block 有 256 个线程，每线程负责 1 个元素，
// 因此每个 block 处理 256 个元素，grid 数量取 (N + 255) / 256（向上取整）。
//
// 参数：
//   x : 输入数组（设备端指针）
//   y : 输出数组
//   N : 元素个数
__global__ void swish_f32_kernel(float *x, float *y, int N) {
  // 全局线程编号：前面 blockIdx.x 个 block 的全部线程 + 自己在 block 内编号
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // 越界保护：N 不一定是启动线程总数的整数倍，多余线程不参与计算
  if (idx < N)
    // 每个元素独立计算：x / (1 + exp(-x))，等价于 x * sigmoid(x)
    y[idx] = swish(x[idx]);
}

// ============================================================================
// Kernel 2：FP32 向量化版（float4 = 4 个 float = 16 字节）
// ============================================================================
// 每个线程连续处理 4 个元素，一次 16 字节的向量 load/store
// 代替 4 次 4 字节访存，从而减少访存指令条数。
//
// 与 elementwise 的 f32x4 不同，Swish 是单输入算子：只读 x 一份数据，
// 没有 b 侧的第二路 load；与 sigmoid 的 f32x4 相比，这里少了“先 clamp”
// 的四行，多了一次乘法（x * sigmoid(x)），指令数量相当。
//
// “向量化”省的只是访存指令：计算仍要按 x/y/z/w 四个分量逐条展开，
// 每个分量都要算一次 expf 和一次除法，属于计算偏重的逐元素算子，
// 因此收益有限（见 docs/kernels/swish/README.md 的性能观察）。
//
// host 端对应启动：block = 256 / 4 = 64 线程，每个 block 仍处理 256 个元素。
__global__ void swish_f32x4_kernel(float *x, float *y, int N) {
  // 基础线程编号 t 对应“第 t 组 4 元素”的起点，真实下标是 4 * t
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    // 把 x[idx..idx+3] 的 16 字节 reinterpret 成 float4，一次读入
    float4 reg_x = FLOAT4(x[idx]);
    // 结果也用 float4 暂存，最后一次性写回
    float4 reg_y;
    // 对 4 个分量分别算 Swish（float4 有 x/y/z/w 四个分量）
    reg_y.x = swish(reg_x.x);
    reg_y.y = swish(reg_x.y);
    reg_y.z = swish(reg_x.z);
    reg_y.w = swish(reg_x.w);
    // 一次 16 字节写回 y[idx..idx+3]
    //
    // 注意（与 elementwise 的差异）：这里只判断了 idx < N，
    // 没有像 elementwise 的 f32x4 那样用 (idx + 3) < N 判断整段 4 个元素
    // 是否齐全，也没有尾部（tail）逐元素回退分支。因此当 N 不是 4 的整数倍时，
    // 最后一个 block 的部分线程会把 idx+1..idx+3 写到 y 的合法范围之外。
    // 本模块基准测试的 S/K 都是 256 的倍数，不会触发该问题；
    // 本次只补充注释，kernel 逻辑保持原样。
    FLOAT4(y[idx]) = reg_y;
  }
}

// ============================================================================
// FP16 辅助函数：half 没有隐式类型提升
// ============================================================================
// 对应关系：/ → __hdiv、+ → __hadd、* → __hmul、-x → __hneg、expf → hexp。
// 常数 1.0 用 __float2half(1.0f) 显式转成 half（字面量参数会被编译器折叠）。
//
// 这里刻意写成 __hmul(x, __hdiv(1, 1 + hexp(-x)))——与 FP32 版的除序相反，
// 但数学上等价（x / d == x * (1 / d)）。写成乘法是因为 half 的除法
// （__hdiv）内部是一段较长的指令序列，反正都要算一次 1 / d，
// 不如用乘法把结果乘回去。
//  FP16
__device__ __forceinline__ half swish_half(half x) {
  return __hmul(x, __hdiv(__float2half(1.0f),
                          __hadd(__float2half(1.0f), hexp(__hneg(x)))));
}

// ============================================================================
// Kernel 3：FP16 标量版
// ============================================================================
// half 是 IEEE 754 半精度浮点，只占 2 字节（float 的一半）。
// 相同显存带宽下，FP16 理论上能搬运两倍元素，代价是精度更低
// （half 约 3~4 位十进制有效数字，保守按约 3 位）。
//
// Swish 在 FP16 下有两处需要注意：
// 1. 分母“吞掉小量”：当 hexp(-x) < 2^-11（约 4.9e-4）时，1 + hexp(-x)
//    会因为 half 的加法舍入直接变成 1.0，于是 sigmoid(x) 被舍入成 1，
//    swish(x) 退化成 x（例如 x = 11 时返回 11.0，与 PyTorch 一致）；
// 2. 负方向饱和：x <= -11.09 时 hexp(-x) 溢出成 inf，__hdiv(1, inf) = 0，
//    结果是 x * 0 = -0.0；而 PyTorch（内部用 float 算 sigmoid）会给出
//    -7.37e-05 这样的小值。绝对误差只有 7e-5，通常无感。
__global__ void swish_f16_kernel(half *x, half *y, int N) {
  // 与 Kernel 1 相同的全局线程编号方式：一线程一元素
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // 越界保护：多余的线程不参与计算
  if (idx < N)
    // 整个 Swish（exp + 加法 + 除法 + 乘法）都封装在 swish_half 里
    y[idx] = swish_half(x[idx]);
}

// ============================================================================
// Kernel 4：FP16 每线程 2 个元素（half2 = 2 个 half = 4 字节）
// ============================================================================
// half2 是 CUDA 内置的 2 元素 half 向量，每线程一次处理 2 个连续元素。
//
// 与 relu 的 f16x2 不同：ReLU 有成对指令 __hmax2，可以把循环步长写成 2；
// Swish 既没有 hexp2 也没有成对的除法，只能逐分量调用 swish_half。
__global__ void swish_f16x2_kernel(half *x, half *y, int N) {
  // 每线程负责以 idx 为起点的 2 个连续 half
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    // 从 x[idx..idx+1] 读 4 字节为一个 half2
    half2 reg_x = HALF2(x[idx]);
    // 结果也用 half2 暂存
    half2 reg_y;
    // half2 只有 x、y 两个分量，分别算 Swish
    reg_y.x = swish_half(reg_x.x);
    reg_y.y = swish_half(reg_x.y);
    // 一次写回 4 字节（2 个 half）
    //
    // 注意（与 elementwise 的差异）：同 Kernel 2，这里只判断了 idx < N，
    // 缺少尾部处理。当 N 不是 2 的整数倍时，会把 idx+1 写到 y 范围之外；
    // 本次只补充注释，kernel 逻辑保持原样。
    HALF2(y[idx]) = reg_y;
  }
}

// ============================================================================
// Kernel 5：FP16 每线程 8 个元素（用 4 次 half2 操作完成，unpack 写法）
// ============================================================================
// 每个线程负责 8 个连续 half，把它们拆成 4 个 half2（reg_x_0..3）处理。
// 相比 f16x2，每个线程做更多工作，能摊薄索引计算等固定开销。
//
// 这里的 “unpack” 指：数据在内存里本就是连续的 8 个 half，
// 本 kernel 不把它整体打包搬运，而是按 half2 逐段读、逐段写。
//
// 注意 4 次 HALF2(x[...]) 读取是无条件执行的，越界判断只出现在写回处。
__global__ void swish_f16x8_kernel(half *x, half *y, int N) {
  // 每线程负责以 idx 为起点的 8 个连续 half
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  // 一次读入 4 个 half2：覆盖 x[idx .. idx+7]
  half2 reg_x_0 = HALF2(x[idx + 0]);
  half2 reg_x_1 = HALF2(x[idx + 2]);
  half2 reg_x_2 = HALF2(x[idx + 4]);
  half2 reg_x_3 = HALF2(x[idx + 6]);
  // 结果也用 4 个 half2 保存
  half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
  // 8 个分量分别算 Swish（4 个 half2 x 2 个分量）
  reg_y_0.x = swish_half(reg_x_0.x);
  reg_y_0.y = swish_half(reg_x_0.y);
  reg_y_1.x = swish_half(reg_x_1.x);
  reg_y_1.y = swish_half(reg_x_1.y);
  reg_y_2.x = swish_half(reg_x_2.x);
  reg_y_2.y = swish_half(reg_x_2.y);
  reg_y_3.x = swish_half(reg_x_3.x);
  reg_y_3.y = swish_half(reg_x_3.y);
  // 结果写回 y[idx .. idx+7]
  //
  // 注意（与 elementwise 的差异）：这里逐个 half2 判断起始下标，
  // 只保证段首不越界。当 N 落在 idx+7 时，(idx + 6) < N 依然成立，
  // 会把 y[idx + 7] 写到 y 范围之外；同时 N 不足 8 个元素的尾部也没有回退分支。
  // 本次只补充注释，kernel 逻辑保持原样。
  if ((idx + 0) < N) {
    HALF2(y[idx + 0]) = reg_y_0;
  }
  if ((idx + 2) < N) {
    HALF2(y[idx + 2]) = reg_y_1;
  }
  if ((idx + 4) < N) {
    HALF2(y[idx + 4]) = reg_y_2;
  }
  if ((idx + 6) < N) {
    HALF2(y[idx + 6]) = reg_y_3;
  }
}

// ============================================================================
// Kernel 6：FP16 每线程 8 个元素（128 位打包版）
// ============================================================================
// 8 个 half 恰好是 16 字节 = 128 位，可以把它们看成一段连续内存，
// 用 reinterpret 成 float4 的方式一次 load/store 128 位，
// 比 Kernel 5 的多次 half2 访存指令更少。
//
// 计算部分：Swish 没有成对指令（不像 ReLU 有 __hmax2），
// 所以循环写成 i++ 逐个 half 调用 swish_half；这一点与 sigmoid 的
// pack 版本写法相同。
//
// 注意（与 elementwise 的差异）：这里只在 (idx + 7) < N 时写回，
// 末尾不足 8 个元素的整段会被直接丢弃（漏算），但不会像 Kernel 2/4/5 那样
// 越界写。本模块基准测试的 S/K 都是 256 的倍数，不会触发该问题；
// 本次只补充注释，kernel 逻辑保持原样。
__global__ void swish_f16x8_pack_kernel(half *x, half *y, int N) {
  // 每线程负责以 idx 为起点的 8 个连续 half
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  // 局部打包数组：8 个 half 正好 128 位。
  // temporary register(memory), .local space in ptx, addressable
  // 这类“可寻址”局部数组在 PTX 中通常对应 .local 空间；但配合下面的
  // #pragma unroll 和编译期常数下标，编译器有机会把它优化到寄存器中，
  // 从而避免真正访问 local memory。
  half pack_x[8], pack_y[8];
  // 把 x[idx..idx+7] 的 16 字节整体 reinterpret 成 float4，
  // 用一条访存指令读入 128 位。
  // reinterpret as float4 and load 128 bits in 1 memory issue.
  LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

#pragma unroll
  for (int i = 0; i < 8; i++) {
    // 逐个 half 算 Swish（没有成对指令可用）
    pack_y[i] = swish_half(pack_x[i]);
  }
  // 结果 pack_y[0..7] 共 128 位，一条 store 指令整体写回 y[idx..idx+7]
  // reinterpret as float4 and store 128 bits in 1 memory issue.
  if ((idx + 7) < N) {
    LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
  }
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

// ----------------------------------------------------------------------------
// 字符串化宏：把函数名转成字符串，供 PYBIND11 注册 Python 函数时使用。
// #str 会把参数按字面内容生成字符串，例如 STRINGFY(swish_f32) 得到
// "swish_f32"。
//
// 紧接着的 TORCH_BINDING_COMMON_EXTENSION 把它包成 m.def(...)，
// 这样 PYTHON 侧就能用 lib.swish_f32 这样的名字直接调用。
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// 类型检查宏：确保传入的 torch::Tensor 是期望的 dtype，
// 否则打印张量信息并抛出运行时异常，避免把错误类型交给 kernel。
// 报错文案是 "values must be ..."。
// ----------------------------------------------------------------------------

// ============================================================================
// host 启动函数生成宏 TORCH_BINDING_SWISH 的说明
//
// 宏参数：
//   packed_type : 命名后缀，如 f32 / f32x4 / f16x8_pack
//   th_type     : 对应的 torch dtype（torch::kFloat32 / torch::kHalf）
//   element_type: kernel 中的 C++ 数据类型（float / half）
//   n_elements  : 每个线程一次处理的元素个数（向量宽度）
//
// 展开后生成形如 swish_f32(torch::Tensor x, torch::Tensor y) 的函数：
//   1. CHECK_TORCH_TENSOR_DTYPE(x, ...)：校验输入张量的 dtype；
//   2. CHECK_TORCH_TENSOR_DTYPE(y, ...)：校验输出张量的 dtype；
//   3. const int ndim = x.dim()：取张量维度，二维 (S, K) 单独走一个分支；
//   4. 计算 dim3 block / dim3 grid，并用 data_ptr() 拿裸指针启动 kernel。
//
// 启动策略：
// - 非二维张量：把各维大小乘起来得到 N，每个 block 固定处理 256 个元素，
//   所以 block = 256 / n_elements，grid = ceil(N / 256)；
// - 二维 (S, K)：每行需要的线程数 K / n_elements 不超过 1024 时，
//   采用“一行一个 block”（grid = S, block = K / n_elements）；
//   行过长则回退到上面的展平策略。
//
// 需要说明的是：宏体内的说明文字不能写成行内注释，否则每行末尾的续行符
// 会被破坏，因此这里把逐条说明集中在宏定义的上方。
// 如果确实要在宏体内部写注释，必须使用 /* ... */ 块注释而不能用 //：
// 宏定义每行以 \ 续行，// 会把该行末尾的续行符一起注释掉，宏定义会当场断开；
// /* ... */ 在预处理阶段被替换为空格，不影响续行。
// ============================================================================

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define TORCH_BINDING_SWISH(packed_type, th_type, element_type, n_elements)    \
  void swish_##packed_type(torch::Tensor x, torch::Tensor y) {                 \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
    const int ndim = x.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= x.size(i);                                                        \
      }                                                                        \
      dim3 block(256 / (n_elements));                                          \
      dim3 grid((N + 256 - 1) / 256);                                          \
      swish_##packed_type##_kernel<<<grid, block>>>(                           \
          reinterpret_cast<element_type *>(x.data_ptr()),                      \
          reinterpret_cast<element_type *>(y.data_ptr()), N);                  \
    } else {                                                                   \
      const int S = x.size(0);                                                 \
      const int K = x.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        dim3 block(K / (n_elements));                                          \
        dim3 grid(S);                                                          \
        swish_##packed_type##_kernel<<<grid, block>>>(                         \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= x.size(i);                                                      \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        swish_##packed_type##_kernel<<<grid, block>>>(                         \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      }                                                                        \
    }                                                                          \
  }

// 实例化 6 个 host 启动函数，分别对应 6 个 kernel：
//   f32         : float，每线程 1 元素
//   f32x4       : float，每线程 4 元素（float4）
//   f16         : half，每线程 1 元素
//   f16x2       : half，每线程 2 元素（half2）
//   f16x8       : half，每线程 8 元素（4 次 half2，unpack）
//   f16x8_pack  : half，每线程 8 元素（128 位打包访存）
TORCH_BINDING_SWISH(f32, torch::kFloat32, float, 1)
TORCH_BINDING_SWISH(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_SWISH(f16, torch::kHalf, half, 1)
TORCH_BINDING_SWISH(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_SWISH(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_SWISH(f16x8_pack, torch::kHalf, half, 8)

// ============================================================================
// PyTorch C++ 扩展模块入口
//
// TORCH_EXTENSION_NAME 由 PyTorch 在编译时注入（这里实际是 swish_lib）。
// m.def 把上面生成的 6 个函数注册为 Python 模块里的同名函数，
// 之后 Python 里就可以直接调用 lib.swish_f32(...) 等。
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 逐个把 C++ host 函数暴露给 Python
  TORCH_BINDING_COMMON_EXTENSION(swish_f32)
  TORCH_BINDING_COMMON_EXTENSION(swish_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(swish_f16)
  TORCH_BINDING_COMMON_EXTENSION(swish_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(swish_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(swish_f16x8_pack)
}
