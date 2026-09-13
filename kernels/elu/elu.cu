// ============================================================================
// elu.cu —— CUDA ELU（Exponential Linear Unit，指数线性单元）教学示例
//
// 本文件实现 y[i] = elu(x[i])，数组长度为 N：
//
//   elu(x) = { x,                    x > 0
//            { alpha * (exp(x) - 1), x <= 0        （本文件 alpha = 1.0，见 ALPHA 宏）
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
// “y[i] = max(0, x[i])” 换成 “y[i] = elu(x[i])”。
// 线程编号、向量化访存、FP16 打包等通用知识不再重复，下面重点说明
// ELU 特有的三点：
//   - 正半轴和 ReLU 一样是恒等映射，负半轴像 sigmoid 一样指数饱和到 -alpha，
//     相当于把两者的优点各取一半（既不平淡无奇地截断，也不会梯度全丢）；
//   - 负半轴要算一次 expf / hexp，所以它比 ReLU 贵、比 gelu 便宜；
//   - 指数只出现在负半轴，因此不需要 sigmoid / gelu 那样的 clamp /
//     溢出保护宏（详见下面的宏定义段）。
//
// 学习重点：
// - 全局线程编号 = blockIdx.x * blockDim.x + threadIdx.x
// - 越界保护：if (idx < N)
// - 三元选择（predicated select）：一条比较 + 两条分支 + 一次选择，
//   与 ReLU 的一次取最大值相比只是多了一次 exp
// - 向量化 / 更宽访存 / FP16 都是逐元素算子的常见带宽优化手段
//
// 文件后半部分是 PyTorch 绑定：把 CUDA kernel 包装成 Python 可调用的函数，
// 供 elu.py 编译与基准测试使用。
// ============================================================================

// ============================================================================
// 【ELU 曲线速览】一张小图看懂这个激活函数
//
//   y = x (x > 0)，y = alpha * (exp(x) - 1) (x <= 0)
//
//  3.0 ┤                          ***
//      ┤                       ***
//  2.0 ┤                    ***
//      ┤                 ***
//  1.0 ┤              ***
//      ┤           ***
//  0.0 ┤**********+
//      ┤      ****
// -0.5 ┤  ****
// -1.0 ┤*····························  ← 渐近线 y = -alpha = -1
//      └──────────────────────────────→ x
//      -3   -2   -1    0    1    2    3
//
// 采样值：-3→-0.9502、-2→-0.8647、-1→-0.6321、0→0、1→1、2→2、3→3
//
// 读图要点：
// - 正半轴是恒等映射（x > 0 时 elu(x) = x），和 ReLU 完全一致：
//   不饱和、梯度恒为 1；
// - 负半轴是指数饱和：x <= 0 时 elu(x) = alpha * (exp(x) - 1)，单调递增、
//   有界，x → -∞ 时极限是 -alpha；但永远取不到 -alpha，
//   所以值域是开区间 (-alpha, +∞)；
// - 在 x = 0 处连续且可导：左导数 exp(0) = 1、右导数 1，两侧相等。
//   elu(0) 走的是 else 分支，算出来是 exp(0) - 1 = 0，与正半轴严格衔接；
// - 负半轴导数 exp(x) 恒大于 0：这与 ReLU 的“负半轴梯度恒为 0”形成关键对比，
//   ELU 不会出现死亡 ReLU（dead ReLU）——负半轴仍然能把梯度回传；
// - 负半轴输出的是负值而不是硬零，缓解了 ReLU 输出恒非负、
//   把下一层偏置整体推偏的问题；
// - 代价是负半轴要算一次指数。因此本模块的性能特征介于 ReLU
//   （纯访存瓶颈）和 sigmoid（计算偏重）之间。
//
// 与 sigmoid / relu 的对照（下文会反复引用这张表）：
//   | 对比项     | sigmoid                    | relu              | elu (alpha=1)          |
//   | 公式       | 1 / (1 + exp(-x))          | max(0, x)         | x>0 ? x : exp(x)-1     |
//   | 值域       | (0, 1)，两端饱和           | [0, +∞)，仅负端   | (-1, +∞)，仅负端饱和   |
//   | x=0 处     | 可导，导数 0.25            | 不可导（左0右1）  | 可导，导数 1           |
//   | 负半轴梯度 | 趋近 0（饱和）             | 恒为 0（死亡）    | exp(x) > 0（衰减但非0）|
//   | 溢出保护   | 需要 clamp                 | 不需要            | 不需要（指数只在负半轴）|
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

// ----------------------------------------------------------------------------
// ALPHA：ELU 的负半轴缩放系数
//
//   elu(x) = x (x > 0)，alpha * (exp(x) - 1) (x <= 0)
//
// 本文件取 alpha = 1.0f，与 PyTorch 的 torch.nn.functional.elu 默认值一致。
// FP32 版本直接使用这个 float 宏；FP16 版本则用 __float2half(ALPHA)
// 现场转换（参数是字面量，编译器会把它折叠成 half 常量，无运行时开销）。
//
// 为什么本文件没有 sigmoid / gelu 那样的 MAX_EXP_* / MIN_EXP_* 溢出保护宏？
// 因为指数只出现在负半轴：
//   - x 是很大的正数：走 x > 0 分支原样返回，不会产生新数值，也就不可能溢出；
//   - x 是很大的负数：exp(x) 下溢成 0，结果是 alpha * (0 - 1) = -alpha，
//     恰好就是数学极限值，也是正确结果；
//   - 即使编译器把三元表达式编译成“两条分支都算、再按谓词选择”，
//     正半轴上算出的 expf(大正数) = inf 也只会被丢掉，不会污染结果。
//
// NaN 语义（与 relu / gelu 对比时值得注意）：分支条件是 x > 0，
// NaN 的比较结果为假，于是走进 exp(NaN) - 1，NaN 被正常传播到输出，
// 与 PyTorch 的 elu 行为一致；而 relu 用的 fmaxf(NaN, 0) = 0
// 会把 NaN 吞掉，gelu 的 fmaxf/fminf 夹取同理。
// ----------------------------------------------------------------------------
// 定义全局 alpha 值
#define ALPHA 1.0f

// ----------------------------------------------------------------------------
// 类型检查宏：确保传入的 torch::Tensor 是期望的 dtype，
// 否则打印张量信息并抛出运行时异常，避免把错误类型交给 kernel。
//
// 注意报错文案是 "Tensor dtype must be ..."，与 relu.cu 里的
// "values must be ..." 略有不同；本次只补充注释，保持原样。
// ----------------------------------------------------------------------------
// 定义 CHECK_TORCH_TENSOR_DTYPE 宏
#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("Tensor dtype must be " #th_type);                \
  }

// ----------------------------------------------------------------------------
// 字符串化宏：把函数名转成字符串，供 PYBIND11 注册 Python 函数时使用。
// #str 会把参数按字面内容生成字符串，例如 STRINGFY(elu_f32) 得到
// "elu_f32"。
// ----------------------------------------------------------------------------
// 定义 TORCH_BINDING_COMMON_EXTENSION 宏
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

// ============================================================================
// ELU 计算函数：为什么写成两个 __device__ __forceinline__ 辅助函数？
// ============================================================================
// ELU 的公式在 FP32 与 FP16 下写法完全不同：
//
//   __device__        ：只能在设备（GPU）代码里调用，不能在 host 上调用；
//   __forceinline__   ：要求编译器强制内联，避免函数调用开销
//                       （逐元素算子每个线程都要调用一次，这点很关键）。
//
// FP32 版直接叫 elu、FP16 版叫 elu_half，与 swish 模块的
// swish / swish_half 命名习惯一致（本次不重命名）。
// ============================================================================
// ELU 计算函数
// FP32

// 三元选择：一条比较 + 两条分支 + 一次选择。
// 注意 expf 是 float 版本（对应 double 版 exp）；参数顺序上先判 x > 0，
// 正半轴直接返回 x，连 exp 都不需要参与最终结果。
__device__ __forceinline__ float elu(float x) {
  return x > 0.f ? x : ALPHA * (expf(x) - 1.f);
}

// FP16

// half 没有隐式类型提升，比较 / 乘法 / 减法 / 指数都要用 half 内建函数：
//   >      → __hgt
//   *      → __hmul
//   -      → __hsub
//   expf   → hexp
// 常数 0、1 与 ALPHA 也用 __float2half 显式转成 half 再参与运算，
// 字面量参数会被编译器直接折叠，不产生运行时开销。
__device__ __forceinline__ half elu_half(half x) {
  return __hgt(x, __float2half(0.f))
             ? x
             : __hmul(__float2half(ALPHA), __hsub(hexp(x), __float2half(1.f)));
}

// ============================================================================
// Kernel 1：FP32 标量版（最朴素写法，作为正确性基准）
// ============================================================================
//
// ---- 调用链：谁触发了这个 kernel？ ----
// elu.py
//   └─ lib.elu_f32(x, y)                            # Python 调用（pybind11）
//       └─ elu_f32(torch::Tensor x, y)              # C++ host 包装函数（宏生成，
//                                       #  运行在 CPU 上）
//           └─ elu_f32_kernel<<<grid, block>>>(...) # CUDA 启动语法
//               └─ cudaLaunchKernel(...)            # nvcc 生成，交给 CUDA runtime
//                   └─ GPU 硬件并行执行本函数体
//
// 关键点：
// - elu_f32 是 CPU 端包装函数，负责类型检查、计算 grid/block；
//   它在源码里由宏 TORCH_BINDING_ELU 生成，直接搜索不到定义。
// - elu_f32_kernel 是 GPU 端 __global__ 函数，
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
// CUDA 核函数
// FP32
__global__ void elu_f32_kernel(float *x, float *y, int N) {
  // 全局线程编号：前面 blockIdx.x 个 block 的全部线程 + 自己在 block 内编号
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // 越界保护：N 不一定是启动线程总数的整数倍，多余线程不参与计算
  if (idx < N)
    // 每个元素独立计算：正半轴返回 x，负半轴算 alpha*(exp(x)-1)。
    // 结果与 PyTorch 的 elu（alpha=1）在 FP32 下一致，NaN 也会被传播。
    y[idx] = elu(x[idx]);
}

// ============================================================================
// Kernel 2：FP32 向量化版（float4 = 4 个 float = 16 字节）
// ============================================================================
// 每个线程连续处理 4 个元素，一次 16 字节的向量 load/store
// 代替 4 次 4 字节访存，从而减少访存指令条数。
//
// 与 elementwise 的 f32x4 不同，ELU 是单输入算子：只读 x 一份数据，
// 没有 b 侧的第二路 load。
//
// “向量化”省的只是访存指令：计算仍要按 x/y/z/w 四个分量逐条展开，
// 每个分量都可能触发一次 expf，属于计算偏重的逐元素算子，
// 因此向量化的收益通常不如 elementwise 加法明显——实测里 f32x4
// 与 f32 基本持平（见 docs/kernels/elu/README.md 的性能观察）。
//
// host 端对应启动：block = 256 / 4 = 64 线程，每个 block 仍处理 256 个元素。
__global__ void elu_f32x4_kernel(float *x, float *y, int N) {
  // 基础线程编号 t 对应“第 t 组 4 元素”的起点，真实下标是 4 * t
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    // 把 x[idx..idx+3] 的 16 字节 reinterpret 成 float4，一次读入
    float4 reg_x = FLOAT4(x[idx]);
    // 结果也用 float4 暂存，最后一次性写回
    float4 reg_y;
    // 对 4 个分量分别算 ELU（float4 有 x/y/z/w 四个分量）
    reg_y.x = elu(reg_x.x);
    reg_y.y = elu(reg_x.y);
    reg_y.z = elu(reg_x.z);
    reg_y.w = elu(reg_x.w);
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

// FP16
// ============================================================================
// Kernel 3：FP16 标量版
// ============================================================================
// half 是 IEEE 754 半精度浮点，只占 2 字节（float 的一半）。
// 相同显存带宽下，FP16 理论上能搬运两倍元素，代价是精度更低
// （half 约 3~4 位十进制有效数字，保守按约 3 位）。
//
// ELU 用 FP16 计算时有两个必须注意的点：
// 1. half 没有 float 那样的隐式类型提升，1.0 / 0.0 这类常数要用
//    __float2half 显式转成 half，否则参与运算的类型不确定；
// 2. 指数要用 half 版本的 hexp，比较要用 __hgt
//    （它们直接处理 16 位 half，不走 float），避免每次计算都插入类型转换。
//
// 负半轴 exp(x)-1 在 x 接近 0 时有相减抵消：exp(x) ≈ 1，两者相减后
// 有效位数会明显减少（FP32 同样存在，只是 FP32 尾数多、影响小）。
__global__ void elu_f16_kernel(half *x, half *y, int N) {
  // 与 Kernel 1 相同的全局线程编号方式：一线程一元素
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // 越界保护：多余的线程不参与计算
  if (idx < N)
    // 整个 ELU（比较 + 负半轴指数）都封装在 elu_half 里
    y[idx] = elu_half(x[idx]);
}

// ============================================================================
// Kernel 4：FP16 每线程 2 个元素（half2 = 2 个 half = 4 字节）
// ============================================================================
// half2 是 CUDA 内置的 2 元素 half 向量，每线程一次处理 2 个连续元素。
//
// 与 relu 的 f16x2 不同：ReLU 有成对指令 __hmax2，可以把循环步长写成 2；
// ELU 没有 hexp2 这样的成对指数指令，只能逐分量调用 elu_half。
__global__ void elu_f16x2_kernel(half *x, half *y, int N) {
  // 每线程负责以 idx 为起点的 2 个连续 half
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    // 从 x[idx..idx+1] 读 4 字节为一个 half2
    half2 reg_x = HALF2(x[idx]);
    // 结果也用 half2 暂存
    half2 reg_y;
    // half2 只有 x、y 两个分量，分别算 ELU
    reg_y.x = elu_half(reg_x.x);
    reg_y.y = elu_half(reg_x.y);
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
__global__ void elu_f16x8_kernel(half *x, half *y, int N) {
  // 每线程负责以 idx 为起点的 8 个连续 half
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  // 一次读入 4 个 half2：覆盖 x[idx .. idx+7]
  half2 reg_x_0 = HALF2(x[idx + 0]);
  half2 reg_x_1 = HALF2(x[idx + 2]);
  half2 reg_x_2 = HALF2(x[idx + 4]);
  half2 reg_x_3 = HALF2(x[idx + 6]);
  // 结果也用 4 个 half2 保存
  half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
  // 8 个分量分别算 ELU（4 个 half2 x 2 个分量）
  reg_y_0.x = elu_half(reg_x_0.x);
  reg_y_0.y = elu_half(reg_x_0.y);
  reg_y_1.x = elu_half(reg_x_1.x);
  reg_y_1.y = elu_half(reg_x_1.y);
  reg_y_2.x = elu_half(reg_x_2.x);
  reg_y_2.y = elu_half(reg_x_2.y);
  reg_y_3.x = elu_half(reg_x_3.x);
  reg_y_3.y = elu_half(reg_x_3.y);
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
// 计算部分的差异值得和其它模块对照：
//   sigmoid : 没有 hexp2，循环只能逐 half 算；
//   relu    : 有 __hmax2，循环写成 i += 2，一次处理一个 half2；
//   elu     : 同样没有成对指令，循环写成 i++，逐个 half 调用 elu_half。
//
// 注意（与 elementwise 的差异）：这里只在 (idx + 7) < N 时写回，
// 末尾不足 8 个元素的整段会被直接丢弃（漏算），但不会像 Kernel 2/4/5 那样
// 越界写。本模块基准测试的 S/K 都是 256 的倍数，不会触发该问题；
// 本次只补充注释，kernel 逻辑保持原样。
__global__ void elu_f16x8_pack_kernel(half *x, half *y, int N) {
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
    // 逐个 half 算 ELU（没有成对指令可用）
    pack_y[i] = elu_half(pack_x[i]);
  }
  // 结果 pack_y[0..7] 共 128 位，一条 store 指令整体写回 y[idx..idx+7]
  // reinterpret as float4 and store 128 bits in 1 memory issue.
  if ((idx + 7) < N) {
    LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
  }
}

// ============================================================================
// host 启动函数生成宏
//
// 宏参数：
//   packed_type : 命名后缀，如 f32 / f32x4 / f16x8_pack
//   th_type     : 对应的 torch dtype（torch::kFloat32 / torch::kHalf）
//   element_type: kernel 中的 C++ 数据类型（float / half）
//   n_elements  : 每个线程一次处理的元素个数（向量宽度）
//
// 展开后生成形如 elu_f32(torch::Tensor x, torch::Tensor y) 的函数：
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
// 需要说明的是：宏体内的说明文字**不能写成行内注释**（那样每行末尾的
// 续行符会被破坏），因此这里把逐条说明集中在宏定义的上方。
// 如果确实要在宏体内部写注释，必须使用 /* ... */ 块注释而不能用 //：
// 宏定义每行以 \ 续行，// 会把该行末尾的续行符一起注释掉，宏定义会当场断开；
// /* ... */ 在预处理阶段被替换为空格，不影响续行。
// ============================================================================
#define TORCH_BINDING_ELU(packed_type, th_type, element_type, n_elements)      \
  void elu_##packed_type(torch::Tensor x, torch::Tensor y) {                   \
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
      elu_##packed_type##_kernel<<<grid, block>>>(                             \
          reinterpret_cast<element_type *>(x.data_ptr()),                      \
          reinterpret_cast<element_type *>(y.data_ptr()), N);                  \
    } else {                                                                   \
      const int S = x.size(0);                                                 \
      const int K = x.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        dim3 block(K / (n_elements));                                          \
        dim3 grid(S);                                                          \
        elu_##packed_type##_kernel<<<grid, block>>>(                           \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= x.size(i);                                                      \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        elu_##packed_type##_kernel<<<grid, block>>>(                           \
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
TORCH_BINDING_ELU(f32, torch::kFloat32, float, 1)
TORCH_BINDING_ELU(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_ELU(f16, torch::kHalf, half, 1)
TORCH_BINDING_ELU(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_ELU(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_ELU(f16x8_pack, torch::kHalf, half, 8)

// ============================================================================
// PyTorch C++ 扩展模块入口
//
// TORCH_EXTENSION_NAME 由 PyTorch 在编译时注入（这里实际是 elu_lib）。
// m.def 把上面生成的 6 个函数注册为 Python 模块里的同名函数，
// 之后 Python 里就可以直接调用 lib.elu_f32(...) 等。
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 逐个把 C++ host 函数暴露给 Python
  TORCH_BINDING_COMMON_EXTENSION(elu_f32)
  TORCH_BINDING_COMMON_EXTENSION(elu_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(elu_f16)
  TORCH_BINDING_COMMON_EXTENSION(elu_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(elu_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(elu_f16x8_pack)
}
