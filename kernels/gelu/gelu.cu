// ============================================================================
// gelu.cu —— CUDA GELU（Gaussian Error Linear Unit，高斯误差线性单元）教学示例
//
// 本文件实现 y[i] = gelu(x[i])，数组长度为 N。PyTorch 支持两种形式
// （torch.nn.GELU 的 approximate 参数）：
//
//   精确式（approximate='none'）：
//     gelu(x) = x * Φ(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
//   tanh 近似（approximate='tanh'）：
//     gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// 其中 Φ(x) 是标准正态分布的累积分布函数（CDF）。
// 本文件默认走 tanh 近似（GELU_OPS / HALF_GELU_OPS 宏指向
// gelu_tanh_approximate），同时保留了 FP32 的精确式实现
// gelu_none_approximate（用 erff），改一行宏定义就能切换。
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
// “y[i] = max(0, x[i])” 换成 “y[i] = gelu(x[i])”。
// 线程编号、向量化访存、FP16 打包等通用知识不再重复，下面重点说明
// GELU 特有的三点：
//   - 公式更长：一次 tanh（或 erf）外面还套着 x 的三次多项式，
//     计算量比 sigmoid / elu 都大；
//   - 必须做溢出保护：half 没有 tanh，这里用 hexp 拼出 tanh，
//     指数必然有溢出边界，因此需要 MAX_EXP_* / MIN_EXP_* 这组 clamp 宏；
//   - FP16 精度陷阱最深：既踩到“相减抵消”的精度坑，
//     也踩到“hexp 溢出成 inf 后 inf/inf = NaN”的正确性坑
//     （详见 docs/kernels/gelu/README.md 的“边界约束”一节）。
//
// 学习重点：
// - 全局线程编号 = blockIdx.x * blockDim.x + threadIdx.x
// - 越界保护：if (idx < N)
// - 先 clamp 再算：exp 系列函数的输入必须先夹到安全区间
// - 宏（GELU_OPS / MAX_EXP_* / SQRT_2_PI）如何用来切换算法与精度档位
//
// 文件后半部分是 PyTorch 绑定：把 CUDA kernel 包装成 Python 可调用的函数，
// 供 gelu.py 编译与基准测试使用。
// ============================================================================

// ============================================================================
// 【GELU 曲线速览】一张小图看懂这个激活函数
//
//   y = x * Φ(x)（tanh 近似画法，x ∈ [-3, 3]）
//
//  4.0 ┤                          ***
//      ┤                       ***
//  3.0 ┤                    ***
//      ┤                 ***
//  2.0 ┤              ***
//      ┤           ***
//  1.0 ┤        ***
//  0.0 ┤*******+                     ← 唯一的零点在 x = 0
//      ┤    ***
// -0.17┤   *                        ← 负向小凹陷，min ≈ -0.1700（x ≈ -0.752）
//      └──────────────────────────────→ x
//      -3   -2   -1   0   1   2   3
//
// 采样值(tanh 近似)：-3→-0.0036、-2→-0.0454、-1→-0.1588、-0.75→-0.1700、
//                    0→0、1→0.8412、2→1.9546、3→2.9964
//
// 读图要点：
// - 正半轴近似恒等映射：x 稍大（约 3 以上）后 Φ(x) ≈ 1，gelu(x) ≈ x；
// - 负半轴不是硬零：x < 0 时输出是小负数，最小值约 -0.1700
//   （出现在 x ≈ -0.752），所以 GELU 不是单调函数；
// - 在 x = 0 处光滑可导（曲线整体无穷阶可导），既没有 ReLU 的
//   “0 点不可导”，也没有 ELU 的“负半轴带一个转折”；
// - 负半轴梯度不为 0：和 ELU 一样避免了死亡 ReLU 问题，
//   但它的负值区间更浅（最深 -0.17，ELU 可以一路滑到 -1）；
// - 计算量最大：一次 tanh（内部还要一次 exp）加若干次乘加，
//   在 relu / elu / sigmoid / gelu / swish 这一族里属于计算偏重的算子。
//
// 与同族算子的对照（下文会反复引用这张表）：
//   | 对比项     | relu            | elu (alpha=1)        | gelu (tanh 近似)      | swish / silu     |
//   | 公式       | max(0, x)       | x>0 ? x : exp(x)-1   | 0.5x(1+tanh(k(...)))  | x/(1+exp(-x))    |
//   | 值域       | [0, +∞)         | (-1, +∞)             | (-0.17, +∞)           | (-0.278, +∞)     |
//   | 单调性     | 单调            | 单调                 | 非单调（负半轴凹陷）  | 非单调           |
//   | x=0 处     | 不可导          | 可导，导数 1         | 光滑可导              | 光滑可导，导数0.5|
//   | 每元素计算 | 一次取最大值    | 一次比较 + 一次 exp  | 多次乘加 + 一次 tanh  | exp + 加/除/乘   |
//   | 溢出保护   | 不需要          | 不需要               | 需要 clamp            | 不需要           |
// ============================================================================

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

// 头文件列表与 elementwise / sigmoid / relu 保持一致（其中若干头文件本模块
// 没有直接使用，保留是为了方便几个模块对照阅读）。
// 本文件额外包含了 cuda_bf16.h 与 cuda_fp8.h：前者是 BFLOAT2 宏所需的
// __nv_bfloat162 类型来源，后者目前并未使用（本文件没有 bf16 / fp8 kernel）。
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
// ----------------------------------------------------------------------------
// exp 溢出保护边界（clamp 的上下限）
//
// GELU 的 tanh 是用 exp 拼出来的（half 没有 tanh）：
//     tanh(t) = (exp(2t) - 1) / (exp(2t) + 1)
// 只要 2t 超过 exp 的溢出阈值，exp(2t) 就变成 inf，分子分母同时变成 inf，
// inf / inf = NaN。所以必须先把输入夹进安全区间，这与 sigmoid 的思路一致。
//
// F32：MAX_EXP_F32 = 88.3762626647949f，是逐元素 sigmoid 常用的保守上界，
//      略小于 ln(FLT_MAX) ≈ 88.723；MIN_EXP_F32 与之对称取负。
// F16：half 的取值范围比 float 窄得多——
//      MAX_EXP_F16 = ln(65504) ≈ 11.09，65504 是 half 的最大规格数；
//      MIN_EXP_F16 = ln(2^-14) ≈ -9.70，2^-14 是 half 的最小规格正数。
//      两个边界不对称，是因为 half 能表示的正数范围本身就偏向 0 一侧。
//
// 【本文件最容易踩的坑】注意下面 kernel 里夹的是输入 x，
// 而真正送进 hexp 的是 2 * inner（inner 含 x 的三次项）：
//     inner = sqrt(2/pi) * (x + 0.044715 * x^3) ≈ 0.797885 * (x + 0.044715x^3)
// 反解 2 * inner = ln(65504) 得到 x ≈ 4.0278：也就是说 FP16 版本只要输入
// 略大于 4.03，hexp 就会溢出并算出 NaN（实测 x=4.0 正常、x=4.1 为 NaN）。
// 这是本模块已识别的问题：本次只补充注释、不改逻辑，
// 修法（改为夹 inner，或把 half 提升成 float 再算 tanh）见
// docs/kernels/gelu/README.md 的“边界约束”一节。
// ----------------------------------------------------------------------------
#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)
// ----------------------------------------------------------------------------
// GELU 系数宏
//
//   M_SQRT2     = sqrt(2)     ≈ 1.41421356
//   M_2_SQRTPI  = 2 / sqrt(pi) ≈ 1.12837917
//   SQRT_2_PI = M_SQRT2 * M_2_SQRTPI * 0.5f ≈ 0.7978845608 = sqrt(2/pi)
//
// 也就是 tanh 近似公式里最前面那个系数（0.7978845608）。
//
// 两个提醒（本次都不修改，仅记录）：
// 1. 紧邻的英文注释把该式写成 sqrt(2 * pi) / 2，但这个式子实际等于
//    sqrt(pi/2) ≈ 1.2533，与代码算出的 0.79788 并不是同一个值
//    （正确的等价写法是 2 / sqrt(2 * pi)）；代码的取值是正确的，
//    只是注释里的代数写反了；
// 2. 宏 SQRT_2_PI 没有加括号，靠“乘法同优先级、从左到右”恰好算对。
//    一旦它出现在优先级更高的表达式里（例如紧跟除号）就会静默出错，
//    规范写法是 #define SQRT_2_PI ((M_SQRT2) * (M_2_SQRTPI) * 0.5f)。
// ----------------------------------------------------------------------------
#define SQRT_2_PI M_SQRT2 *M_2_SQRTPI * 0.5f
#define HALF_1 __float2half(1.0f)
#define HALF_2 __float2half(2.0f)
#define HALF_DIV2 __float2half(0.5f)
// HALF_SQRT_2_PI：half 版的 sqrt(2/pi) 系数。
// 用“先由 FP32 字面量转出 half、再做 half 乘法”的方式构造，
// 与 HALF_DIV2 = __float2half(0.5f) 配合得到 0.7979 附近的值。
// HALF_V_APP = 0.044715 是 tanh 近似公式里三次项的系数。
// （上面的英文注释想表达的“用 sqrt(2*pi)/2 去掉自定义 gelu 与 PyTorch gelu
//   之间的误差”这一结论是对的：half 计算路径不同会带来末位差异。）
// to clear the error among self defined gelu and pytorch gelu. Calculate
// $\sqrt{\frac{\pi}{2}}$ by $\sqrt{2 * \pi} / 2$
#define HALF_SQRT_2_PI                                                         \
  __float2half(M_SQRT2) * __float2half(M_2_SQRTPI) * HALF_DIV2
#define HALF_V_APP __float2half(0.044715f)

// ----------------------------------------------------------------------------
// 算法选择开关
//
//   HALF_GELU_OPS : FP16 走哪个函数（half 版本目前只有 tanh 近似一种）
//   GELU_OPS      : FP32 走哪个函数，可切换为：
//                   gelu_tanh_approximate（tanh 近似，默认）
//                   gelu_none_approximate（用 erff 的精确式）
//
// 改这两行就能在“精度 / 速度”之间切换，不需要动任何 kernel 代码。
// 注意 PyTorch 的 torch.nn.GELU 默认是 approximate='none'（erf 精确式），
// 要让对照结果对齐，必须显式写 torch.nn.GELU("tanh")，gelu.py 就是这么做的。
// ----------------------------------------------------------------------------
#define HALF_GELU_OPS gelu_tanh_approximate
#define GELU_OPS gelu_tanh_approximate

// ============================================================================
// GELU 计算函数：三个重载 / 变体
//
//   gelu_tanh_approximate(half)  —— FP16，tanh 近似（用 hexp 拼 tanh）
//   gelu_tanh_approximate(float) —— FP32，tanh 近似（直接调用 tanhf）
//   gelu_none_approximate(float) —— FP32，精确式（用 erff）
//
// 三个函数都用 __inline__ __device__ 修饰：
//   __device__ ：只能在设备（GPU）代码里调用，不能在 host 上调用；
//   __inline__ ：建议编译器内联，避免函数调用开销
//                （逐元素算子每个线程都要调用一次，这点很关键）。
//
// 之所以要分 FP16 / FP32 两个版本，是因为 half 没有隐式类型提升：
// half 的乘法、加法、指数都必须用内建函数（__hmul / __hadd / hexp ...），
// 常数也要用 __float2half 显式转换。
// ============================================================================
// There is no half presicion operation like sinh, cosh, tanh. [Half Math
// Functions](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__FUNCTIONS.html#group__CUDA__MATH____HALF__FUNCTIONS)
// $$ tanh(x) = \frac{exp^{2x} - 1}{exp^{2x} + 1}$$
// But ops above will introduce error.
// pytorch transform type while do tanh operator which include in the
// [pytorch/c10/util/BFloat16-math.h](https://github.com/pytorch/pytorch/blob/main/c10/util/BFloat16-math.h)
__inline__ __device__ half gelu_tanh_approximate(half x) {
  // 三次项 x^3：half 的乘法可以直接写 *，因为编译选项里恢复了
  // __CUDA_NO_HALF_OPERATORS__（见 gelu.py 的 -U 选项）
  half x_cube = x * x * x;
  // compute mid value : inner = 0.7978845608 * (x + 0.044715 * x * x * x)
  // inner 是 tanh 的实参；注意它含三次项，增长远比 x 快
  half inner = HALF_SQRT_2_PI * (x + HALF_V_APP * x_cube);
  // compute tanh
  // 用 (e^{2t} - 1) / (e^{2t} + 1) 拼出 tanh：
  //   - inner 接近 0 时，hexp(2*inner) ≈ 1，分子是两个接近的数相减，
  //     有效位数大量丢失——这就是“half 版误差比 PyTorch 大”的根源；
  //   - inner 偏大时，hexp(2*inner) 溢出成 inf，分子分母都变成 inf，
  //     inf / inf = NaN——这就是 FP16 版本在 x ≳ 4.03 时输出 NaN 的原因；
  // 本次只补充注释，kernel 逻辑保持原样。
  return HALF_DIV2 * x *
         (HALF_1 +
          ((hexp(inner * HALF_2) - HALF_1) / (hexp(inner * HALF_2) + HALF_1)));
}

__inline__ __device__ float gelu_tanh_approximate(float x) {
  // FP32 版直接用 CUDA 的 tanhf，不存在上面的两个问题；
  // SQRT_2_PI * (x + 0.044715 * x^3) 就是 tanh 近似的实参。
  // --use_fast_math 会让 tanhf 走到更快、精度略低的实现路径。
  return 0.5f * x * (1.0f + tanhf(SQRT_2_PI * (x + 0.044715f * x * x * x)));
}

// 精确式（approximate='none'）：0.5 * x * (1 + erf(x / sqrt(2)))，
// 其中 M_SQRT1_2 = 1/sqrt(2) ≈ 0.7071。把 GELU_OPS 宏改成本函数即可切换。
__inline__ __device__ float gelu_none_approximate(float x) {
  return x * 0.5 * (1 + erff(x * M_SQRT1_2));
}

// ============================================================================
// Kernel 1：FP32 标量版（最朴素写法，作为正确性基准）
// ============================================================================
//
// ---- 调用链：谁触发了这个 kernel？ ----
// gelu.py
//   └─ lib.gelu_f32(x, y)                            # Python 调用（pybind11）
//       └─ gelu_f32(torch::Tensor x, y)              # C++ host 包装函数（宏生成，
//                                         #  运行在 CPU 上）
//           └─ gelu_f32_kernel<<<grid, block>>>(...) # CUDA 启动语法
//               └─ cudaLaunchKernel(...)             # nvcc 生成，交给 CUDA runtime
//                   └─ GPU 硬件并行执行本函数体
//
// 关键点：
// - gelu_f32 是 CPU 端包装函数，负责类型检查、计算 grid/block；
//   它在源码里由宏 TORCH_BINDING_GELU 生成，直接搜索不到定义。
// - gelu_f32_kernel 是 GPU 端 __global__ 函数，
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
// FP32
// GELU tanh approximate: x, y:x 0.5 * x
// * (1.0 + tanh(0.7978845608 * x * (1.0 + 0.044715 * x * x))) grid(N/256),
// block(K=256)
__global__ void gelu_f32_kernel(float *x, float *y, int N) {
  // 全局线程编号：前面 blockIdx.x 个 block 的全部线程 + 自己在 block 内编号
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // 越界保护：N 不一定是启动线程总数的整数倍，多余线程不参与计算
  if (idx < N) {
    // 先 clamp 再算：把 v 夹在 [MIN_EXP_F32, MAX_EXP_F32] 内，
    // 保证后续 tanhf 的实参不会落到无穷/NaN 的危险区。
    // fminf / fmaxf 是 float 版本（写成 fmin / fmax 会提升成 double，更慢）。
    // 注意 fmaxf(NaN, MIN_EXP_F32) = MIN_EXP_F32（IEEE maxNum 语义），
    // 所以 NaN 输入会被“洗”成 -88.376 并最终输出 0，与 PyTorch 不一致；
    // 同时 |x| > 88.376 的输入会被截断（gelu_f32(100) = 88.376），
    // 本次只补充注释，kernel 逻辑保持原样。
    float v = fminf(fmaxf(x[idx], MIN_EXP_F32), MAX_EXP_F32);
    // GELU_OPS 默认展开为 gelu_tanh_approximate（FP32 tanh 近似）
    y[idx] = GELU_OPS(v);
  }
}

// ============================================================================
// Kernel 2：FP32 向量化版（float4 = 4 个 float = 16 字节）
// ============================================================================
// 每个线程连续处理 4 个元素，一次 16 字节的向量 load/store
// 代替 4 次 4 字节访存，从而减少访存指令条数。
//
// 与 elementwise 的 f32x4 不同，GELU 是单输入算子：只读 x 一份数据。
//
// GELU 的算术强度高（一次 tanh 内部就含一次 exp），访存指令数不是瓶颈，
// 所以向量化那点收益容易被“逐分量展开的冗长计算”吃掉——
// 实测 f32x4 甚至比标量版略慢（见 docs/kernels/gelu/README.md 的性能观察）。
//
// 与其它 kernel 的写法差异：这里没有在 load 之前做越界判断，
// 而是先无条件读入 float4，最后只在写回时检查 (idx + 0) < N。
//
// host 端对应启动：block = 256 / 4 = 64 线程，每个 block 仍处理 256 个元素。
// GELU tanh approximate; Vec4
// grid(N/256), block(256/4)
__global__ void gelu_f32x4_kernel(float *x, float *y, int N) {
  // 基础线程编号 t 对应“第 t 组 4 元素”的起点，真实下标是 4 * t
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  // 无条件读入 16 字节：x[idx..idx+3]。
  // 注意这里没有 idx < N 的保护，N 不是 4 的倍数时会越界读（见文末注释）。
  float4 reg_x = FLOAT4(x[idx]);
  // 结果也用 float4 暂存，最后一次性写回
  float4 reg_y;

  // 对 4 个分量分别 clamp，理由同标量版 Kernel 1
  reg_x.x = fminf(fmaxf(reg_x.x, MIN_EXP_F32), MAX_EXP_F32);
  reg_x.y = fminf(fmaxf(reg_x.y, MIN_EXP_F32), MAX_EXP_F32);
  reg_x.z = fminf(fmaxf(reg_x.z, MIN_EXP_F32), MAX_EXP_F32);
  reg_x.w = fminf(fmaxf(reg_x.w, MIN_EXP_F32), MAX_EXP_F32);

  // 对 4 个分量分别算 GELU（float4 有 x/y/z/w 四个分量）
  reg_y.x = GELU_OPS(reg_x.x);
  reg_y.y = GELU_OPS(reg_x.y);
  reg_y.z = GELU_OPS(reg_x.z);
  reg_y.w = GELU_OPS(reg_x.w);

  // 一次 16 字节写回 y[idx..idx+3]
  //
  // 注意（与 elementwise 的差异）：这里只判断了 (idx + 0) < N，
  // 没有像 elementwise 的 f32x4 那样用 (idx + 3) < N 判断整段 4 个元素
  // 是否齐全，也没有尾部（tail）逐元素回退分支。因此当 N 不是 4 的整数倍时，
  // load 会越界读 1~3 个 float，而 idx < N 仍成立时还会越界写 1~3 个 float。
  // 本模块基准测试的 S/K 都是 256 的倍数，不会触发该问题；
  // 本次只补充注释，kernel 逻辑保持原样。
  if ((idx + 0) < N) {
    FLOAT4(y[idx]) = reg_y;
  }
}

// ============================================================================
// Kernel 3：FP16 标量版
// ============================================================================
// half 是 IEEE 754 半精度浮点，只占 2 字节（float 的一半）。
// 相同显存带宽下，FP16 理论上能搬运两倍元素，代价是精度更低
// （half 约 3~4 位十进制有效数字，保守按约 3 位）。
//
// GELU 用 FP16 计算时有三个必须注意的点：
// 1. half 没有隐式类型提升，常数要用 __float2half 显式转成 half；
// 2. 绝对值保护要用 __hmin / __hmax（直接比较 16 位 half，不走 float）；
// 3. 计算走 HALF_GELU_OPS，也就是用 hexp 拼 tanh 的近似实现——
//    它既有“相减抵消”的精度损失，也有“hexp 溢出 → inf/inf = NaN”
//    的正确性问题（输入 x ≳ 4.03 时触发，见文件开头与
//    docs/kernels/gelu/README.md 的“边界约束”）。
// FP16
// GELU approximate: x, y:x 0.5 * x *
// (1.0 + tanh(0.7978845608 (x + 0.044715 * x * x * x))) Vec4
__global__ void gelu_f16_kernel(half *x, half *y, int N) {
  // 与 Kernel 1 相同的全局线程编号方式：一线程一元素
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // 越界保护：多余的线程不参与计算
  if (idx < N) {
    // 先取出输入，再做 half 版 clamp（上下界不对称，见宏定义段的说明）
    half v = x[idx];
    v = __hmin(__hmax(v, MIN_EXP_F16), MAX_EXP_F16);

    // HALF_GELU_OPS 默认展开为 gelu_tanh_approximate(half)
    y[idx] = HALF_GELU_OPS(v);
  }
}

// ============================================================================
// Kernel 4：FP16 每线程 2 个元素（half2 = 2 个 half = 4 字节）
// ============================================================================
// half2 是 CUDA 内置的 2 元素 half 向量，每线程一次处理 2 个连续元素。
//
// 与 relu 的 f16x2 不同：ReLU 有成对指令 __hmax2，可以把循环步长写成 2；
// GELU 没有成对指令（没有 htanh2），只能逐分量计算。
//
// 写法上与其它模块的差异：这里没有先判断 idx < N 再 load，
// 而是无条件读入 half2，最后只在写回时检查 (idx + 0) < N。
__global__ void gelu_f16x2_kernel(half *x, half *y, int N) {
  // 每线程负责以 idx 为起点的 2 个连续 half
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

  // 无条件读入 4 字节：x[idx..idx+1]
  half2 reg_x = HALF2(x[idx]);
  // 结果也用 half2 暂存
  half2 reg_y;
  // 先对 2 个分量分别 clamp
  reg_x.x = __hmin(__hmax(reg_x.x, MIN_EXP_F16), MAX_EXP_F16);
  reg_x.y = __hmin(__hmax(reg_x.y, MIN_EXP_F16), MAX_EXP_F16);

  // 再对 2 个分量分别算 GELU
  reg_y.x = HALF_GELU_OPS(reg_x.x);
  reg_y.y = HALF_GELU_OPS(reg_x.y);
  // 一次写回 4 字节（2 个 half）
  //
  // 注意（与 elementwise 的差异）：这里只判断了 (idx + 0) < N，
  // 缺少尾部处理。当 N 是奇数时，load 会越界读 1 个 half，
  // 且 idx < N 仍成立时还会越界写 1 个 half；
  // 本次只补充注释，kernel 逻辑保持原样。
  if ((idx + 0) < N) {
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
// 排版上的小细节：计算时把结果又写回了 reg_x_*（复用了输入寄存器），
// 而不是像其它模块那样用单独的 reg_y_* 接收结果——本次只补充注释，
// 不改逻辑。
//
// 注意 4 次 HALF2(x[...]) 读取是无条件执行的，越界判断只出现在写回处。
// unpack f16x8
__global__ void gelu_f16x8_kernel(half *x, half *y, int N) {
  // 每线程负责以 idx 为起点的 8 个连续 half
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

  // 一次读入 4 个 half2：覆盖 x[idx .. idx+7]
  half2 reg_x_0 = HALF2(x[idx + 0]);
  half2 reg_x_1 = HALF2(x[idx + 2]);
  half2 reg_x_2 = HALF2(x[idx + 4]);
  half2 reg_x_3 = HALF2(x[idx + 6]);

  // 先对 8 个分量逐个 clamp
  reg_x_0.x = __hmin(__hmax(reg_x_0.x, MIN_EXP_F16), MAX_EXP_F16);
  reg_x_0.y = __hmin(__hmax(reg_x_0.y, MIN_EXP_F16), MAX_EXP_F16);
  reg_x_1.x = __hmin(__hmax(reg_x_1.x, MIN_EXP_F16), MAX_EXP_F16);
  reg_x_1.y = __hmin(__hmax(reg_x_1.y, MIN_EXP_F16), MAX_EXP_F16);
  reg_x_2.x = __hmin(__hmax(reg_x_2.x, MIN_EXP_F16), MAX_EXP_F16);
  reg_x_2.y = __hmin(__hmax(reg_x_2.y, MIN_EXP_F16), MAX_EXP_F16);
  reg_x_3.x = __hmin(__hmax(reg_x_3.x, MIN_EXP_F16), MAX_EXP_F16);
  reg_x_3.y = __hmin(__hmax(reg_x_3.y, MIN_EXP_F16), MAX_EXP_F16);

  // 原本准备用于存放结果的 4 个 half2（本 kernel 实际把结果写回了 reg_x_*）
  half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;

  // 再对 8 个分量逐个算 GELU
  reg_x_0.x = HALF_GELU_OPS(reg_x_0.x);
  reg_x_0.y = HALF_GELU_OPS(reg_x_0.y);
  reg_x_1.x = HALF_GELU_OPS(reg_x_1.x);
  reg_x_1.y = HALF_GELU_OPS(reg_x_1.y);
  reg_x_2.x = HALF_GELU_OPS(reg_x_2.x);
  reg_x_2.y = HALF_GELU_OPS(reg_x_2.y);
  reg_x_3.x = HALF_GELU_OPS(reg_x_3.x);
  reg_x_3.y = HALF_GELU_OPS(reg_x_3.y);

  // 结果写回 y[idx .. idx+7]
  //
  // 注意（与 elementwise 的差异）：这里逐个 half2 判断起始下标，
  // 只保证段首不越界。当 N 落在 idx+7 时，(idx + 6) < N 依然成立，
  // 会把 y[idx + 7] 写到 y 范围之外；同时 N 不足 8 个元素的尾部也没有回退分支。
  // 本次只补充注释，kernel 逻辑保持原样。
  if ((idx + 0) < N) {
    HALF2(y[idx + 0]) = reg_x_0;
  }
  if ((idx + 2) < N) {
    HALF2(y[idx + 2]) = reg_x_1;
  }
  if ((idx + 4) < N) {
    HALF2(y[idx + 4]) = reg_x_2;
  }
  if ((idx + 6) < N) {
    HALF2(y[idx + 6]) = reg_x_3;
  }
}

// ============================================================================
// Kernel 6：FP16 每线程 8 个元素（128 位打包版）
// ============================================================================
// 8 个 half 恰好是 16 字节 = 128 位，可以把它们看成一段连续内存，
// 用 reinterpret 成 float4 的方式一次 load/store 128 位，
// 比 Kernel 5 的多次 half2 访存指令更少，实测也是 FP16 里最快的。
//
// 计算部分的差异值得和其它模块对照：
//   sigmoid : 没有 hexp2，循环只能 i++ 逐个 half 算；
//   relu    : 有 __hmax2，循环写成 i += 2，一次处理一个 half2；
//   gelu    : 同样没有成对指令，循环写成 ++i，逐个 half 算 tanh 近似。
//
// 注意（与 elementwise 的差异）：这里只在 (idx + 7) < N 时写回，
// 末尾不足 8 个元素的整段会被直接丢弃（漏算），但不会像 Kernel 2/4/5 那样
// 越界写。本模块基准测试的 S/K 都是 256 的倍数，不会触发该问题；
// 本次只补充注释，kernel 逻辑保持原样。
// pack f16x8
__global__ void gelu_f16x8_pack_kernel(half *x, half *y, int N) {
  // 每线程负责以 idx 为起点的 8 个连续 half
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

  // temporary register(memory), .local space in ptx, addressable
  // 局部打包数组：8 个 half 正好 128 位。这类“可寻址”局部数组在 PTX 中
  // 通常对应 .local 空间；但配合下面的 #pragma unroll 和编译期常数下标，
  // 编译器有机会把它优化到寄存器中，从而避免真正访问 local memory。
  half pack_x[8], pack_y[8]; // 8x16 bits=128 bits.
  // 把 x[idx..idx+7] 的 16 字节整体 reinterpret 成 float4，
  // 用一条访存指令读入 128 位。
  // reinterpret as float4 and load 128 bits in 1 memory issue.
  LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); // load 128 bits

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    // 逐个 half：先 clamp 再算 GELU（没有成对指令可用）
    half v = __hmin(__hmax(pack_x[i], MIN_EXP_F16), MAX_EXP_F16);
    pack_y[i] = HALF_GELU_OPS(v);
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
// #str 会把参数按字面内容生成字符串，例如 STRINGFY(gelu_f32) 得到
// "gelu_f32"。
//
// 紧接着的 TORCH_BINDING_COMMON_EXTENSION 把它包成 m.def(...)，
// 这样 Python 侧就能用 lib.gelu_f32 这样的名字直接调用。
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// 类型检查宏：确保传入的 torch::Tensor 是期望的 dtype，
// 否则打印张量信息并抛出运行时异常，避免把错误类型交给 kernel。
// 报错文案是 "values must be ..."。
// ----------------------------------------------------------------------------

// ============================================================================
// host 启动函数生成宏 TORCH_BINDING_GELU 的说明
//
// 宏参数：
//   packed_type : 命名后缀，如 f32 / f32x4 / f16x8_pack
//   th_type     : 对应的 torch dtype（torch::kFloat32 / torch::kHalf）
//   element_type: kernel 中的 C++ 数据类型（float / half）
//   n_elements  : 每个线程一次处理的元素个数（向量宽度）
//
// 展开后生成形如 gelu_f32(torch::Tensor x, torch::Tensor y) 的函数：
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

#define TORCH_BINDING_GELU(packed_type, th_type, element_type, n_elements)     \
  void gelu_##packed_type(torch::Tensor x, torch::Tensor y) {                  \
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
      gelu_##packed_type##_kernel<<<grid, block>>>(                            \
          reinterpret_cast<element_type *>(x.data_ptr()),                      \
          reinterpret_cast<element_type *>(y.data_ptr()), N);                  \
    } else {                                                                   \
      const int S = x.size(0);                                                 \
      const int K = x.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        dim3 block(K / (n_elements));                                          \
        dim3 grid(S);                                                          \
        gelu_##packed_type##_kernel<<<grid, block>>>(                          \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= x.size(i);                                                      \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        gelu_##packed_type##_kernel<<<grid, block>>>(                          \
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
TORCH_BINDING_GELU(f32, torch::kFloat32, float, 1)
TORCH_BINDING_GELU(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_GELU(f16, torch::kHalf, half, 1)
TORCH_BINDING_GELU(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_GELU(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_GELU(f16x8_pack, torch::kHalf, half, 8)

// ============================================================================
// PyTorch C++ 扩展模块入口
//
// TORCH_EXTENSION_NAME 由 PyTorch 在编译时注入（这里实际是 gelu_lib）。
// m.def 把上面生成的 6 个函数注册为 Python 模块里的同名函数，
// 之后 Python 里就可以直接调用 lib.gelu_f32(...) 等。
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 逐个把 C++ host 函数暴露给 Python
  TORCH_BINDING_COMMON_EXTENSION(gelu_f32)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f16)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f16x8_pack)
}
