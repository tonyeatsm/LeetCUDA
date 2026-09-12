// ============================================================================
// relu.cu —— CUDA ReLU（修正线性单元）教学示例
//
// 本文件实现 y[i] = max(0, x[i])，数组长度为 N。
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
// 与 elementwise / sigmoid 模块的关系：
// 本文件的 6 个版本与 kernels/elementwise/elementwise.cu、
// kernels/sigmoid/sigmoid.cu 的 6 个版本一一对应，区别只是把
// “c[i] = a[i] + b[i]” / “y[i] = sigmoid(x[i])” 换成 “y[i] = max(0, x[i])”。
// 线程编号、向量化访存、FP16 打包等通用知识不再重复，下面重点说明
// ReLU 特有的两点：
//   - 计算极轻：只有一次取最大值，没有 exp、没有除法，所以
//     sigmoid 必须做的 clamp / 溢出保护（MAX_EXP_* 宏）在这里完全不需要；
//   - 与 sigmoid 的模板差异：ReLU 有成对的 __hmax2 指令，f16x8_pack 的循环
//     因此可以写成 i += 2；而 sigmoid 没有 hexp2，只能逐个 half 算。
//
// 学习重点：
// - 全局线程编号 = blockIdx.x * blockDim.x + threadIdx.x
// - 越界保护：if (idx < N)
// - 逐元素算子可以“一个线程算多个元素”，用更宽的访存换更少的访存指令
// - 当计算量小到可以忽略时，耗时基本由访存决定，
//   因此本模块也是观察显存带宽上限的最佳样本
//
// 文件后半部分是 PyTorch 绑定：把 CUDA kernel 包装成 Python 可调用的函数，
// 供 relu.py 编译与基准测试使用。
// ============================================================================

// ============================================================================
// 【ReLU 曲线速览】一张小图看懂这个激活函数
//
//   y = max(0, x)
//
//  3.0 ┤                          ***
//      ┤                       ***
//  2.0 ┤                    ***
//      ┤                 ***
//  1.0 ┤              ***
//      ┤           ***
//  0.0 ┤***********+
//      └──────────────────────────────→ x
//      -3     -2    -1    0    1    2    3
//
// 采样值：-3→0、-1→0、0→0、1→1、2→2、3→3
// （负半轴被“整流”成 0，正半轴是恒等映射，这就是 Rectified 的来历）
//
// 读图要点：
// - 分段线性：x > 0 时 relu(x) = x，x <= 0 时 relu(x) = 0；
// - 值域是 [0, +∞)，不像 sigmoid 那样是 (0, 1) 的有界区间，两端都不饱和：
//   正半轴导数恒为 1，梯度不会随 x 增大而衰减，所以深网络里比 sigmoid
//   更不容易梯度消失——这是 ReLU 成为现代网络隐藏层默认选择的主因；
// - 在 x = 0 处不可导（左导 0、右导 1），工程上取次梯度，常用 0；
// - 负半轴导数恒为 0：若某个神经元的输出长期落在负半轴，它的梯度恒 0、
//   权重不再更新，即所谓“死亡 ReLU（dead ReLU）”，这也是 leaky ReLU /
//   GELU / SiLU 等变体出现的动机；
// - 计算量极低：整个算子只有一次比较/取最大值，没有 exp、没有除法，
//   所以它既是很好的入门示例，也可以当作同类逐元素算子的性能上限参照。
//
// 与 sigmoid 的对照（下文会反复引用这张表）：
//   | 对比项     | sigmoid                      | relu                         |
//   | 公式       | 1 / (1 + exp(-x))            | max(0, x)                    |
//   | 值域       | (0, 1)，两端饱和             | [0, +∞)，正半轴不饱和        |
//   | 每元素计算 | 一次 exp + 一次加法 + 一次除法 | 一次取最大值               |
//   | 溢出保护   | 必须 clamp 后再 exp          | 不需要（见下面的宏定义段）   |
//   | 成对指令   | 无 hexp2，只能逐分量算       | 有 __hmax2，一次算 2 个 half |
// ============================================================================

// 头文件列表与 elementwise / sigmoid 保持一致（其中若干头文件本模块没有
// 直接使用，保留是为了方便几个模块对照阅读）。
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

// WARP_SIZE：一个 warp（线程束）固定有 32 个线程，
// 现代 NVIDIA GPU 调度和执行指令的最小单位。
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
// 说明：INT4 与 BFLOAT2 本模块没有使用，保留是为了和 elementwise / sigmoid
// 的宏定义段保持一致；BFLOAT2 依赖 cuda_bf16.h，本文件没有包含该头文件，
// 但这两个宏从未被展开，所以不影响编译。
// ----------------------------------------------------------------------------
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
// 128 位 = 16 字节访存，本文件用它一次读写 8 个 half 组成的打包数据
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
// ----------------------------------------------------------------------------
// 为什么本文件没有 sigmoid 那样的 MAX_EXP_* 边界宏？
//
// sigmoid 要先算 exp(-x) 再取倒数，exp 的参数太大就会溢出成 inf，
// 所以它必须先把 x 夹在 [MIN_EXP_F32, MAX_EXP_F32]（FP16 是 F16 版本）之内。
//
// ReLU 只需要一次取最大值：
//   - x 是很大的正数：原样返回，不会产生新的数值，也就不会溢出；
//   - x 是很大的负数：被截断成 0；
//   - x 是 ±inf：得到 +inf / 0。
// 结果一定落在 [0, +∞) 内，全程不需要夹取，因此不需要任何边界宏。
//
// 唯一需要留意的是 NaN 的处理，见 relu_f32_kernel 内的说明。
// ----------------------------------------------------------------------------

// ============================================================================
// Kernel 1：FP32 标量版（最朴素写法，作为正确性基准）
// ============================================================================
//
// ---- 调用链：谁触发了这个 kernel？ ----
// relu.py
//   └─ lib.relu_f32(x, y)                              # Python 调用（pybind11）
//       └─ relu_f32(torch::Tensor x, y)                # C++ host 包装函数（宏生成，
//                                         #  运行在 CPU 上）
//           └─ relu_f32_kernel<<<grid, block>>>(...)   # CUDA 启动语法
//               └─ cudaLaunchKernel(...)               # nvcc 生成，交给 CUDA runtime
//                   └─ GPU 硬件并行执行本函数体
//
// 关键点：
// - relu_f32 是 CPU 端包装函数，负责类型检查、计算 grid/block；
//   它在源码里由宏 TORCH_BINDING_RELU 生成，直接搜索不到定义。
// - relu_f32_kernel 是 GPU 端 __global__ 函数，
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
__global__ void relu_f32_kernel(float *x, float *y, int N) {
  // 全局线程编号：前面 blockIdx.x 个 block 的全部线程 + 自己在 block 内编号
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // 越界保护：N 不一定是启动线程总数的整数倍，多余线程不参与计算
  if (idx < N)
    // fmaxf 是 float 版本的取最大值函数，不能写成 fmax（那是 double 版本，
    // 会把参数提升成 double 再比较，明显更慢）。
    // 参数顺序写成「0.0f 在前」只是习惯写法：除 NaN 外 fmaxf 是对称的。
    // NaN 语义：IEEE 754-2008 的 maxNum 规定“一个参数是 NaN 时返回另一个
    // 参数”，所以 fmaxf(NaN, 0.0f) == 0.0f，即本 kernel 会把 NaN 输入清成 0；
    // 而 PyTorch 的 torch.relu 会把 NaN 传播到输出。本模块基准测试的输入由
    // torch.randn 生成（有限值），不会触发该差异。本次只补充注释，
    // kernel 逻辑保持原样。
    y[idx] = fmaxf(0.0f, x[idx]);
}

// ============================================================================
// Kernel 2：FP32 向量化版（float4 = 4 个 float = 16 字节）
// ============================================================================
// 每个线程连续处理 4 个元素，一次 16 字节的向量 load/store
// 代替 4 次 4 字节访存，从而减少访存指令条数。
//
// 与 elementwise 的 f32x4 不同，ReLU 是单输入算子：只读 x 一份数据，
// 没有 b 侧的第二路 load。
//
// ReLU 的计算只有 4 次 fmaxf，比 sigmoid 的 4 次 expf 轻得多，
// 所以“省下的访存指令”更容易直接体现在耗时上；反过来，当规模大到已经
// 吃满显存带宽时，向量化也就不会再有额外收益（见 README 的实测数据）。
//
// host 端对应启动：block = 256 / 4 = 64 线程，每个 block 仍处理 256 个元素。
__global__ void relu_f32x4_kernel(float *x, float *y, int N) {
  // 基础线程编号 t 对应“第 t 组 4 元素”的起点，真实下标是 4 * t
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    // 把 x[idx..idx+3] 的 16 字节 reinterpret 成 float4，一次读入
    float4 reg_x = FLOAT4(x[idx]);
    // 结果也用 float4 暂存，最后一次性写回
    float4 reg_y;
    // 对 4 个分量分别取最大值（float4 有 x/y/z/w 四个分量）
    reg_y.x = fmaxf(0.0f, reg_x.x);
    reg_y.y = fmaxf(0.0f, reg_x.y);
    reg_y.z = fmaxf(0.0f, reg_x.z);
    reg_y.w = fmaxf(0.0f, reg_x.w);
    // 一次 16 字节写回 y[idx..idx+3]
    //
    // 注意（与 elementwise 的差异）：这里只判断了 (idx + 0) < N，
    // 没有像 elementwise 的 f32x4 那样用 (idx + 3) < N 判断整段 4 个元素是否
    // 齐全，也没有尾部（tail）逐元素回退分支。因此当 N 不是 4 的整数倍时，
    // 最后一个 block 的部分线程会把 idx+1..idx+3 写到 y 的合法范围之外。
    // 本模块基准测试的 S/K 都是 256 的倍数，不会触发该问题；
    // 本次只补充注释，kernel 逻辑保持原样。
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
// 与 sigmoid 的 f16 版本相比，这里少了一处常量复用：
// sigmoid 要预先算好 const half f = __float2half(1.0f)，是因为它留在例子里的
// 常量 1 要在分子分母各出现一次；ReLU 只需要一个 0，而
// __float2half(0.0f) 的参数是字面量，编译器会直接折叠成 half 常量，
// 不会产生运行时的类型转换开销。
//
// 取最大值要用 half 版本的 __hmax（直接比较 16 位 half），写成 fmaxf
// 会先把 half 提升到 float、算完再转回来，多出两条转换指令。
__global__ void relu_f16_kernel(half *x, half *y, int N) {
  // 与 Kernel 1 相同的全局线程编号方式：一线程一元素
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // 越界保护：多余的线程不参与计算
  if (idx < N)
    // __hmax 比较两个 half 标量并返回较大者，较大的那个写回 y
    y[idx] = __hmax(__float2half(0.0f), x[idx]);
}

// ============================================================================
// Kernel 4：FP16 每线程 2 个元素（half2 = 2 个 half = 4 字节）
// ============================================================================
// half2 是 CUDA 内置的 2 元素 half 向量，每线程一次处理 2 个连续元素。
//
// ReLU 其实有成对指令 __hmax2（一次比较一个 half2 里的 2 个 half），
// 但本 kernel 为了与 elementwise / sigmoid 的 f16x2 保持一致的“逐分量展开”
// 写法，仍按 reg_y.x / reg_y.y 两行分别计算，成对指令的用法放在 Kernel 6。
__global__ void relu_f16x2_kernel(half *x, half *y, int N) {
  // 每线程负责以 idx 为起点的 2 个连续 half
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    // 从 x[idx..idx+1] 读 4 字节为一个 half2
    half2 reg_x = HALF2(x[idx]);
    // 先读一次 y 只是为了拿到一个 half2 类型的变量；
    // 紧随其后的两行会把 reg_y 的两个分量全部覆盖，所以这次读取属于
    // 冗余访存（保留原样，本次只补充注释）。
    half2 reg_y = HALF2(y[idx]);
    // half2 只有 x、y 两个分量，分别取最大值
    reg_y.x = __hmax(__float2half(0.0f), reg_x.x);
    reg_y.y = __hmax(__float2half(0.0f), reg_x.y);
    // 一次写回 4 字节（2 个 half）
    //
    // 注意（与 elementwise 的差异）：同 Kernel 2，这里只判断了 (idx + 0) < N，
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
__global__ void relu_f16x8_kernel(half *x, half *y, int N) {
  // 每线程负责以 idx 为起点的 8 个连续 half
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  // 一次读入 4 个 half2：覆盖 x[idx .. idx+7]
  half2 reg_x_0 = HALF2(x[idx + 0]);
  half2 reg_x_1 = HALF2(x[idx + 2]);
  half2 reg_x_2 = HALF2(x[idx + 4]);
  half2 reg_x_3 = HALF2(x[idx + 6]);
  // 结果也用 4 个 half2 保存
  half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
  // 8 个分量分别取最大值（4 个 half2 x 2 个分量）
  reg_y_0.x = __hmax(__float2half(0.0f), reg_x_0.x);
  reg_y_0.y = __hmax(__float2half(0.0f), reg_x_0.y);
  reg_y_1.x = __hmax(__float2half(0.0f), reg_x_1.x);
  reg_y_1.y = __hmax(__float2half(0.0f), reg_x_1.y);
  reg_y_2.x = __hmax(__float2half(0.0f), reg_x_2.x);
  reg_y_2.y = __hmax(__float2half(0.0f), reg_x_2.y);
  reg_y_3.x = __hmax(__float2half(0.0f), reg_x_3.x);
  reg_y_3.y = __hmax(__float2half(0.0f), reg_x_3.y);
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
// 与 elementwise / sigmoid 的 f16x8_pack 一样，这里的 “pack” 指的是把局部
// 数组整体当作 128 位向量搬运；计算部分仍是按 half 粒度处理。
//
// 计算部分的差异值得和 sigmoid 对照：
//   sigmoid : 没有 hexp2，循环只能 for (int i = 0; i < 8; ++i) 逐个 half 算；
//   relu    : 有 __hmax2，循环写成 for (int i = 0; i < 8; i += 2)，
//             一次处理一个 half2（2 个 half），8 个元素只需 4 次成对指令。
//
// 注意（与 elementwise 的差异）：这里只在 (idx + 7) < N 时写回，
// 末尾不足 8 个元素的整段会被直接丢弃（漏算），但不会像 Kernel 2/4/5 那样
// 越界写。本模块基准测试的 S/K 都是 256 的倍数，不会触发该问题；
// 本次只补充注释，kernel 逻辑保持原样。
__global__ void relu_f16x8_pack_kernel(half *x, half *y, int N) {
  // 每线程负责以 idx 为起点的 8 个连续 half
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  // 常数 0 的 half2 形式：__hmax2 要求两个操作数同类型，
  // 这里显式构造一次并在循环中复用（两个 __float2half 都是字面量，会被折叠）
  const half2 z2 = {__float2half(0.0f), __float2half(0.0f)};
  // 局部打包数组：8 个 half 正好 128 位。
  // temporary register(memory), .local space in ptx, addressable
  // 这类“可寻址”局部数组在 PTX 中通常对应 .local 空间；但配合下面的
  // #pragma unroll 和编译期常数下标，编译器有机会把它优化到寄存器中，
  // 从而避免真正访问 local memory。
  half pack_x[8], pack_y[8]; // 8x16 bits=128 bits.
  // 把 x[idx..idx+7] 的 16 字节整体 reinterpret 成 float4，
  // 用一条访存指令读入 128 位。
  // reinterpret as float4 and load 128 bits in 1 memory issue.
  LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); // load 128 bits

#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    // __hmax2 for half2 x 4
    // 每个 i 处理 pack_x[i] 与 pack_x[i+1] 两个 half：
    // HALF2(pack_x[i]) 把这两个 half 重解释成一个 half2，
    // __hmax2 一条指令对 2 个分量同时取最大值，
    // 结果写回 pack_y[i]、pack_y[i+1]
    HALF2(pack_y[i]) = __hmax2(HALF2(pack_x[i]), z2);
  }
  // 结果 pack_y[0..7] 共 128 位，一条 store 指令整体写回 y[idx..idx+7]
  // reinterpret as float4 and store 128 bits in 1 memory issue.
  if ((idx + 7) < N) {
    LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
  }
}

// ----------------------------------------------------------------------------
// 字符串化宏：把函数名转成字符串，供 PYBIND11 注册 Python 函数时使用。
// #str 会把参数按字面内容生成字符串，例如 STRINGFY(relu_f32) 得到
// "relu_f32"。
// ----------------------------------------------------------------------------
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

// ----------------------------------------------------------------------------
// 类型检查宏：确保传入的 torch::Tensor 是期望的 dtype，
// 否则打印张量信息并抛出运行时异常，避免把错误类型交给 kernel。
// ----------------------------------------------------------------------------
#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
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
// 展开后生成形如 relu_f32(torch::Tensor x, torch::Tensor y) 的函数：
//   1. 校验输入、输出张量的 dtype；
//   2. 根据张量形状计算 block / grid；
//   3. 用 data_ptr() 拿裸指针并启动对应 kernel。
//
// 启动策略：
// - 非二维张量：展平成 N，每个 block 固定处理 256 个元素，
//   所以 block = 256 / n_elements，grid = ceil(N / 256)；
// - 二维 (S, K)：每行需要的线程数不超过 1024 时采用“一行一个 block”
//   （grid = S, block = K / n_elements）；行过长则回退到上面的展平策略。
//
// 下面的注释放在宏体内部，必须用 /* ... */ 而不能用 //：
// 宏定义每行以 \ 续行，// 会把该行末尾的续行符一起注释掉，宏定义会当场断开；
// /* ... */ 在预处理阶段被替换为空格，不影响续行。
// ============================================================================
#define TORCH_BINDING_RELU(packed_type, th_type, element_type, n_elements)     \
  void relu_##packed_type(torch::Tensor x, torch::Tensor y) {                  \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))  /* 校验输入 x 的 dtype */          \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))  /* 校验输出 y 的 dtype */          \
    const int ndim = x.dim();  /* 张量维度：二维 (S, K) 单独分支，其余按一维展平 */ \
    if (ndim != 2) {  /* 非二维：展平成 N，按每 block 256 个元素启动 */        \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= x.size(i);                                                        \
      }                                                                        \
      dim3 block(256 / (n_elements));  /* 展平策略：每 block 处理 256 个元素 */ \
      dim3 grid((N + 256 - 1) / 256);  /* grid 向上取整 */                     \
      relu_##packed_type##_kernel<<<grid, block>>>(  /* 启动 kernel：每线程处理 n_elements 个元素 */ \
          reinterpret_cast<element_type *>(x.data_ptr()),                      \
          reinterpret_cast<element_type *>(y.data_ptr()), N);                  \
    } else {                                                                   \
      const int S = x.size(0);  /* 二维：S 是行数 */                           \
      const int K = x.size(1);  /* 二维：K 是每行元素数，N = S * K */          \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {  /* 每行线程数不超过 1024，可一行一个 block */ \
        dim3 block(K / (n_elements));  /* block 覆盖一整行 */                  \
        dim3 grid(S);  /* grid 大小 = 行数 */                                  \
        relu_##packed_type##_kernel<<<grid, block>>>(                          \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= x.size(i);                                                      \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        relu_##packed_type##_kernel<<<grid, block>>>(                          \
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
TORCH_BINDING_RELU(f32, torch::kFloat32, float, 1)
TORCH_BINDING_RELU(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_RELU(f16, torch::kHalf, half, 1)
TORCH_BINDING_RELU(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_RELU(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_RELU(f16x8_pack, torch::kHalf, half, 8)

// ============================================================================
// PyTorch C++ 扩展模块入口
//
// TORCH_EXTENSION_NAME 由 PyTorch 在编译时注入（这里实际是 relu_lib）。
// m.def 把上面生成的 6 个函数注册为 Python 模块里的同名函数，
// 之后 Python 里就可以直接调用 lib.relu_f32(...) 等。
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 逐个把 C++ host 函数暴露给 Python
  TORCH_BINDING_COMMON_EXTENSION(relu_f32)
  TORCH_BINDING_COMMON_EXTENSION(relu_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(relu_f16)
  TORCH_BINDING_COMMON_EXTENSION(relu_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(relu_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(relu_f16x8_pack)
}
