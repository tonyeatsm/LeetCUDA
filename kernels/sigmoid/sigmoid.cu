// ============================================================================
// sigmoid.cu —— CUDA Sigmoid（S 型激活函数）教学示例
//
// 本文件实现 y[i] = 1 / (1 + exp(-x[i]))，数组长度为 N。
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
// 与 elementwise 模块的关系：
// 本文件的 6 个版本与 kernels/elementwise/elementwise.cu 的 6 个版本一一对应，
// 区别只是把 “c[i] = a[i] + b[i]” 换成 “y[i] = sigmoid(x[i])”，
// 且 sigmoid 是单输入算子（只读 x 一份数据，不像加法要读 a、b 两份）。
// 线程编号、向量化访存、FP16 打包等通用知识不再重复，下面重点说明
// sigmoid 特有的两点：
//   - 数值稳定性：exp 的溢出边界与 clamp（先夹取再取指数）；
//   - 数据类型精度：half 运算必须用内建函数，常数 1.0 要用 __float2half 转。
//
// 学习重点：
// - 全局线程编号 = blockIdx.x * blockDim.x + threadIdx.x
// - 越界保护：if (idx < N)
// - exp 溢出保护：把 x 限制在 exp 不溢出的区间，再做 1/(1+exp(-x))
// - 向量化 / 更宽访存 / FP16 都是逐元素算子的常见带宽优化手段
//
// 文件后半部分是 PyTorch 绑定：把 CUDA kernel 包装成 Python 可调用的函数，
// 供 sigmoid.py 编译与基准测试使用。
// ============================================================================

// ============================================================================
// 【Sigmoid 曲线速览】一张小图看懂这个激活函数
//
//   y = 1 / (1 + e^(-x))
//
//  1.0 ┤                     **********
//      ┤                  ***
//      ┤                **
//  0.5 ┤               +
//      ┤             **
//      ┤          ***
//  0.0 ┤**********
//       └──────────────────────────────→ x
//       -6      -3     0      3       6
//
// 采样值：-6→0.0025、-3→0.0474、0→0.5000、3→0.9526、6→0.9975
//
// 读图要点：
// - S 形、单调递增：左边贴近 0、右边贴近 1，中间平滑过渡；
// - 值域恒为 (0, 1) 开区间，输出可解释为概率 / 门控开关；
// - 关于 (0, 0.5) 中心对称：sigmoid(-x) = 1 - sigmoid(x)；
//   导数 sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))，在 x = 0 处最大（0.25），
//   所以中心附近最敏感、近似线性；
// - 两端饱和：|x| 超过约 8 后输出几乎不再变化，这就是下文 kernel 先做 clamp(截断)
//   再算 exp 的依据——饱和后的钳位不改变结果，却能避免 exp 溢出；
// - 作为激活函数：它给网络引入非线性（否则多层线性变换叠加仍是线性变换）；
//   缺点是两端导数趋近 0，深网络容易梯度消失，现代隐藏层多用 ReLU，
//   但二分类输出层、LSTM 门控等仍常见。
// ============================================================================

// 头文件列表与 elementwise.cu 保持一致（其中若干头文件本模块没有直接使用，
// 保留是为了方便两个模块对照阅读）。
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
// ----------------------------------------------------------------------------
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
// 128 位 = 16 字节访存，本文件用它一次读写 8 个 half 组成的打包数据
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
// ----------------------------------------------------------------------------
// exp 溢出保护边界
//
// sigmoid 要算 exp(-x)，而 exp 的参数太大会溢出成 inf，太小会下溢成 0，
// 所以本文件先把输入 x 夹在 [MIN_EXP, MAX_EXP] 区间内，再去算指数。
//
// F32：MAX_EXP_F32 = 88.3762626647949f ≈ ln(2^127.5)，
//      略小于 ln(FLT_MAX) ≈ 88.723，是逐元素 sigmoid 常用的保守上界；
//      exp 的参数落在 ±88.376 内时结果一定是有限值，不会 inf 也不会 0。
// F16：half 的取值范围比 float 窄得多——
//      MAX_EXP_F16 = ln(65504) ≈ 11.09，65504 是 half 的最大规格数；
//      MIN_EXP_F16 = ln(2^-14) ≈ -9.70，2^-14 是 half 的最小规格正数。
//      两个边界不对称，是因为 half 能表示的正数范围本身就偏向 0 一侧。
//
// sigmoid 在两端是饱和的：|x| 超出这些边界后，结果在对应精度下已经是 1.0
// 或 0.0，因此钳位不会改变最终可表示的结果。
// ----------------------------------------------------------------------------
#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)

// ============================================================================
// Kernel 1：FP32 标量版（最朴素写法，作为正确性基准）
// ============================================================================
//
// ---- 调用链：谁触发了这个 kernel？ ----
// sigmoid.py
//   └─ lib.sigmoid_f32(x, y)                              # Python 调用（pybind11）
//       └─ sigmoid_f32(torch::Tensor x, y)                # C++ host 包装函数（宏生成，
//                                         #  运行在 CPU 上）
//           └─ sigmoid_f32_kernel<<<grid, block>>>(...)   # CUDA 启动语法
//               └─ cudaLaunchKernel(...)                  # nvcc 生成，交给 CUDA runtime
//                   └─ GPU 硬件并行执行本函数体
//
// 关键点：
// - sigmoid_f32 是 CPU 端包装函数，负责类型检查、计算 grid/block；
//   它在源码里由宏 TORCH_BINDING_SIGMOID 生成，直接搜索不到定义。
// - sigmoid_f32_kernel 是 GPU 端 __global__ 函数，
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
__global__ void sigmoid_f32_kernel(float *x, float *y, int N) {
  // 全局线程编号：前面 blockIdx.x 个 block 的全部线程 + 自己在 block 内编号
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // 越界保护：N 不一定是启动线程总数的整数倍，多余线程不参与计算
  if (idx < N) {
    // 1) 数值保护：把 v 夹在 exp 不会溢出的区间内。
    //    fminf / fmaxf 是 float 版本，不能写成 fmin / fmax（那是 double 版本，
    //    会把参数提升成 double 再比较，明显更慢）。
    float v = x[idx];
    v = fminf(fmaxf(v, MIN_EXP_F32), MAX_EXP_F32);
    // 2) 计算 sigmoid：clamp 之后 exp(-v) 一定是有限非零值，
    //    1 + exp(-v) 不会变成 inf，结果稳定落在 (0, 1) 内。
    //    expf 同样是 float 版本（对应 double 版 exp）。
    y[idx] = 1.0f / (1.0f + expf(-v));
  }
}

// ============================================================================
// Kernel 2：FP32 向量化版（float4 = 4 个 float = 16 字节）
// ============================================================================
// 每个线程连续处理 4 个元素，一次 16 字节的向量 load/store
// 代替 4 次 4 字节访存，从而减少访存指令条数。
//
// 与 elementwise 的 f32x4 不同，sigmoid 是单输入算子：只读 x 一份数据，
// 没有 b 侧的第二路 load。
//
// 注意“向量化”省的只是访存指令：计算仍要按 x/y/z/w 四个分量逐条展开，
// 而且每个分量都要做一次 expf，属于计算偏重的逐元素算子，
// 因此向量化带来的加速比通常不如 elementwise 加法明显。
//
// host 端对应启动：block = 256 / 4 = 64 线程，每个 block 仍处理 256 个元素。
__global__ void sigmoid_f32x4_kernel(float *x, float *y, int N) {
  // 基础线程编号 t 对应“第 t 组 4 元素”的起点，真实下标是 4 * t
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  // 把 x[idx..idx+3] 的 16 字节 reinterpret 成 float4，一次读入
  float4 reg_x = FLOAT4(x[idx]);
  float4 reg_y;

  // 对 4 个分量分别做 clamp，理由同标量版 Kernel 1
  reg_x.x = fminf(fmaxf(reg_x.x, MIN_EXP_F32), MAX_EXP_F32);
  reg_x.y = fminf(fmaxf(reg_x.y, MIN_EXP_F32), MAX_EXP_F32);
  reg_x.z = fminf(fmaxf(reg_x.z, MIN_EXP_F32), MAX_EXP_F32);
  reg_x.w = fminf(fmaxf(reg_x.w, MIN_EXP_F32), MAX_EXP_F32);

  // 对 4 个分量分别计算 sigmoid（float4 有 x/y/z/w 四个分量）
  reg_y.x = 1.0f / (1.0f + expf(-reg_x.x));
  reg_y.y = 1.0f / (1.0f + expf(-reg_x.y));
  reg_y.z = 1.0f / (1.0f + expf(-reg_x.z));
  reg_y.w = 1.0f / (1.0f + expf(-reg_x.w));

  // 一次 16 字节写回 y[idx..idx+3]
  //
  // 注意（与 elementwise 的差异）：这里只判断了 (idx + 0) < N，
  // 没有像 elementwise 的 f32x4 那样用 (idx + 3) < N 判断整段 4 个元素是否
  // 齐全，也没有尾部（tail）逐元素回退分支。因此当 N 不是 4 的整数倍时，
  // 最后一个 block 的部分线程会把 idx+1..idx+3 写到 y 的合法范围之外。
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
// sigmoid 用 FP16 计算时有两个必须注意的点：
// 1. half 没有 float 那样的隐式类型提升，1.0 这类常数要用
//    __float2half(1.0f) 显式转成 half，否则参与运算的类型不确定；
// 2. 指数要用 half 版本 hexp，绝对值保护用 __hmin / __hmax
//    （它们直接比较 16 位 half，不走 float），避免每次计算都插入类型转换。
__global__ void sigmoid_f16_kernel(half *x, half *y, int N) {
  // 与 Kernel 1 相同的全局线程编号方式：一线程一元素
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // 预先把常数 1.0 转成 half 存进 f 复用，省掉每个元素重复转换的开销
  const half f = __float2half(1.0f);
  // 越界保护：多余的线程不参与计算
  if (idx < N) {
    // clamp：把 v 限制在 hexp 不溢出的区间（边界见文件开头的 F16 说明）
    half v = x[idx];
    v = __hmin(__hmax(v, MIN_EXP_F16), MAX_EXP_F16);
    // f / (f + hexp(-v))：用 half 的指数与除法完成 sigmoid
    y[idx] = f / (f + hexp(-v));
  }
}

// ============================================================================
// Kernel 4：FP16 每线程 2 个元素（half2 = 2 个 half = 4 字节）
// ============================================================================
// half2 是 CUDA 内置的 2 元素 half 向量，每线程一次处理 2 个连续元素。
//
// 与 elementwise 的 f16x2 有一点不同：加法有现成的 __hadd2 成对指令，
// 而指数没有对应的 hexp2，所以两个分量仍要分别调用 hexp。
__global__ void sigmoid_f16x2_kernel(half *x, half *y, int N) {
  // 每线程负责以 idx 为起点的 2 个连续 half
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  const half f = __float2half(1.0f);
  // 从 x[idx..idx+1] 读 4 字节为一个 half2
  half2 reg_x = HALF2(x[idx]);
  half2 reg_y;
  // half2 只有 x、y 两个分量，分别 clamp
  reg_x.x = __hmin(__hmax(reg_x.x, MIN_EXP_F16), MAX_EXP_F16);
  reg_x.y = __hmin(__hmax(reg_x.y, MIN_EXP_F16), MAX_EXP_F16);

  // 再分别计算 sigmoid（没有成对的 hexp2，只能按分量调用 hexp）
  reg_y.x = f / (f + hexp(-reg_x.x));
  reg_y.y = f / (f + hexp(-reg_x.y));

  // 一次写回 4 字节（2 个 half）
  //
  // 注意（与 elementwise 的差异）：同 Kernel 2，这里只判断了 (idx + 0) < N，
  // 缺少尾部处理。当 N 不是 2 的整数倍时，会把 idx+1 写到 y 范围之外；
  // 本次只补充注释，kernel 逻辑保持原样。
  if ((idx + 0) < N) {
    HALF2(y[idx]) = reg_y;
  }
}

// ============================================================================
// Kernel 5：FP16 每线程 8 个元素（用 4 次 half2 操作完成，unpack 写法）
// ============================================================================
// 每个线程负责 8 个连续 half，把它们拆成 4 个 half2（reg_*_0..3）处理。
// 相比 f16x2，每个线程做更多工作，能摊薄索引计算等固定开销。
//
// 这里的 “unpack” 指：数据在内存里本就是连续的 8 个 half，
// 本 kernel 不把它整体打包搬运，而是按 half2 逐段读、逐段写。
__global__ void sigmoid_f16x8_kernel(half *x, half *y, int N) {
  // 每线程负责以 idx 为起点的 8 个连续 half
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  // 预先把常数 1.0 转成 half
  const half f = __float2half(1.0f);

  // 一次读入 4 个 half2：覆盖 x[idx .. idx+7]
  half2 reg_x_0 = HALF2(x[idx + 0]);
  half2 reg_x_1 = HALF2(x[idx + 2]);
  half2 reg_x_2 = HALF2(x[idx + 4]);
  half2 reg_x_3 = HALF2(x[idx + 6]);

  // 8 个分量分别 clamp（4 个 half2 × 2 个分量）
  reg_x_0.x = __hmin(__hmax(reg_x_0.x, MIN_EXP_F16), MAX_EXP_F16);
  reg_x_0.y = __hmin(__hmax(reg_x_0.y, MIN_EXP_F16), MAX_EXP_F16);
  reg_x_1.x = __hmin(__hmax(reg_x_1.x, MIN_EXP_F16), MAX_EXP_F16);
  reg_x_1.y = __hmin(__hmax(reg_x_1.y, MIN_EXP_F16), MAX_EXP_F16);
  reg_x_2.x = __hmin(__hmax(reg_x_2.x, MIN_EXP_F16), MAX_EXP_F16);
  reg_x_2.y = __hmin(__hmax(reg_x_2.y, MIN_EXP_F16), MAX_EXP_F16);
  reg_x_3.x = __hmin(__hmax(reg_x_3.x, MIN_EXP_F16), MAX_EXP_F16);
  reg_x_3.y = __hmin(__hmax(reg_x_3.y, MIN_EXP_F16), MAX_EXP_F16);

  // 结果也用 4 个 half2 保存
  half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;

  // 8 个分量分别计算 sigmoid
  reg_y_0.x = f / (f + hexp(-reg_x_0.x));
  reg_y_0.y = f / (f + hexp(-reg_x_0.y));
  reg_y_1.x = f / (f + hexp(-reg_x_1.x));
  reg_y_1.y = f / (f + hexp(-reg_x_1.y));
  reg_y_2.x = f / (f + hexp(-reg_x_2.x));
  reg_y_2.y = f / (f + hexp(-reg_x_2.y));
  reg_y_3.x = f / (f + hexp(-reg_x_3.x));
  reg_y_3.y = f / (f + hexp(-reg_x_3.y));

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
// 与 elementwise 的 f16x8_pack 一样，这里的 “pack” 指的是把局部数组整体
// 当作 128 位向量搬运；计算部分仍是逐个 half 处理。
//
// 注意（与 elementwise 的差异）：这里只在 (idx + 7) < N 时写回，
// 末尾不足 8 个元素的整段会被直接丢弃（漏算），但不会像 Kernel 2/4/5 那样
// 越界写。本模块基准测试的 S/K 都是 256 的倍数，不会触发该问题；
// 本次只补充注释，kernel 逻辑保持原样。
__global__ void sigmoid_f16x8_pack_kernel(half *x, half *y, int N) {
  // 每线程负责以 idx 为起点的 8 个连续 half
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  // 预先把常数 1.0 转成 half
  const half f = __float2half(1.0f);
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

  // 循环展开：每轮处理 1 个 half，共 8 轮
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    // 每个元素先 clamp，再算 f / (f + hexp(-v))
    half v = __hmin(__hmax(pack_x[i], MIN_EXP_F16), MAX_EXP_F16);
    pack_y[i] = f / (f + hexp(-v));
  }
  // 结果 pack_y[0..7] 共 128 位，一条 store 指令整体写回 y[idx..idx+7]
  // reinterpret as float4 and store 128 bits in 1 memory issue.
  if ((idx + 7) < N) {
    LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
  }
}

// ----------------------------------------------------------------------------
// 字符串化宏：把函数名转成字符串，供 PYBIND11 注册 Python 函数时使用。
// #str 会把参数按字面内容生成字符串，例如 STRINGFY(sigmoid_f32) 得到
// "sigmoid_f32"。
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
// 展开后生成形如 sigmoid_f32(torch::Tensor x, torch::Tensor y) 的函数：
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
#define TORCH_BINDING_SIGMOID(packed_type, th_type, element_type, n_elements)  \
  void sigmoid_##packed_type(torch::Tensor x, torch::Tensor y) {               \
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
      sigmoid_##packed_type##_kernel<<<grid, block>>>(  /* 启动 kernel：每线程处理 n_elements 个元素 */ \
          reinterpret_cast<element_type *>(x.data_ptr()),                      \
          reinterpret_cast<element_type *>(y.data_ptr()), N);                  \
    } else {                                                                   \
      const int S = x.size(0);  /* 二维：S 是行数 */                           \
      const int K = x.size(1);  /* 二维：K 是每行元素数，N = S * K */          \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {  /* 每行线程数不超过 1024，可一行一个 block */ \
        dim3 block(K / (n_elements));  /* block 覆盖一整行 */                  \
        dim3 grid(S);  /* grid 大小 = 行数 */                                  \
        sigmoid_##packed_type##_kernel<<<grid, block>>>(                       \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= x.size(i);                                                      \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        sigmoid_##packed_type##_kernel<<<grid, block>>>(                       \
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
TORCH_BINDING_SIGMOID(f32, torch::kFloat32, float, 1)
TORCH_BINDING_SIGMOID(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_SIGMOID(f16, torch::kHalf, half, 1)
TORCH_BINDING_SIGMOID(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_SIGMOID(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_SIGMOID(f16x8_pack, torch::kHalf, half, 8)

// ============================================================================
// PyTorch C++ 扩展模块入口
//
// TORCH_EXTENSION_NAME 由 PyTorch 在编译时注入（这里实际是 sigmoid_lib）。
// m.def 把上面生成的 6 个函数注册为 Python 模块里的同名函数，
// 之后 Python 里就可以直接调用 lib.sigmoid_f32(...) 等。
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 逐个把 C++ host 函数暴露给 Python
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f32)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16x8_pack)
}
