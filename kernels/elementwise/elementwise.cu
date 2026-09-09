// ============================================================================
// elementwise.cu —— CUDA 逐元素加法（elementwise add）教学示例
//
// 本文件实现 c[i] = a[i] + b[i]，数组长度为 N。
// 所有 kernel 都把输入输出当作“一维数组”看待，二维 (S, K) 矩阵在启动端
// 展平成 N = S * K。
//
// 文件中包含 6 个计算版本：
//   1. f32        : FP32 标量版（每线程 1 个元素）
//   2. f32x4      : FP32 向量化版（每线程 4 个元素，16 字节访存）
//   3. f16        : FP16 标量版（half，每线程 1 个元素）
//   4. f16x2      : FP16 每线程 2 个元素（half2，4 字节访存）
//   5. f16x8      : FP16 每线程 8 个元素（4 次 half2 操作）
//   6. f16x8_pack : FP16 每线程 8 个元素（按 128 位打包成 1 次访存）
//
// 学习重点：
// - 全局线程编号 = blockIdx.x * blockDim.x + threadIdx.x
// - 越界保护：if (idx < N)
// - 向量化 / 更宽访存 / FP16 都是 elementwise kernel 常见的带宽优化手段
//
// 文件后半部分是 PyTorch 绑定：把 CUDA kernel 包装成 Python 可调用的函数，
// 供 elementwise.py 编译与基准测试使用。
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

// WARP_SIZE：一个 warp（线程束）固定有 32 个线程，
// 现代 NVIDIA GPU 调度和执行指令的最小单位。
#define WARP_SIZE 32

// ----------------------------------------------------------------------------
// 访存“重解释（reinterpret）”宏
//
// GPU 一次访存指令能搬移的位宽有限。把一段连续内存 reinterpret 成更宽的
// CUDA 内置向量类型，可以让编译器生成更宽的 load/store 指令。
//
// 用法示例：FLOAT4(a[idx]) 把 &a[idx] 起的 16 字节当成一个 float4 读写。
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

// ============================================================================
// Kernel 1：FP32 标量版（最朴素写法，作为正确性基准）
// ============================================================================
//
// ---- 调用链：谁触发了这个 kernel？ ----
// elementwise.py
//   └─ lib.elementwise_add_f32(a, b, c)            # Python 调用（pybind11）
//       └─ elementwise_add_f32(...)                # C++ host 包装函数（宏生成，
//                                                  #  运行在 CPU 上）
//           └─ elementwise_add_f32_kernel<<<grid, block>>>(...)  # CUDA 启动语法
//               └─ cudaLaunchKernel(...)           # nvcc 生成，交给 CUDA runtime
//                   └─ GPU 硬件并行执行本函数体
//
// 关键点：
// - elementwise_add_f32 是 CPU 端包装函数，负责类型检查、计算 grid/block；
//   它在源码里由宏 TORCH_BINDING_ELEM_ADD 生成，直接搜索不到定义。
// - elementwise_add_f32_kernel 是 GPU 端 __global__ 函数，
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
//   a, b : 两个输入数组（设备端指针）
//   c    : 输出数组
//   N    : 元素个数
__global__ void elementwise_add_f32_kernel(float *a, float *b, float *c,
                                           int N) {
  // 全局线程编号：前面 blockIdx.x 个 block 的全部线程 + 自己在 block 内编号
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // 越界保护：N 不一定是启动线程总数的整数倍，多余线程不参与计算
  if (idx < N)
    c[idx] = a[idx] + b[idx];
}

// ============================================================================
// Kernel 2：FP32 向量化版（float4 = 4 个 float = 16 字节）
// ============================================================================
// 每个线程连续处理 4 个元素，一次 16 字节的向量 load/store
// 代替 4 次 4 字节访存，从而减少访存指令条数。
//
// host 端对应启动：block = 256 / 4 = 64 线程，每个 block 仍处理 256 个元素。
__global__ void elementwise_add_f32x4_kernel(float *a, float *b, float *c,
                                             int N) {
  // 基础线程编号 t 对应“第 t 组 4 元素”的起点，真实下标是 4 * t
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  // 主体：从 idx 开始连续 4 个元素都有效时才走向量化分支
  if ((idx + 3) < N) {
    // 把 a[idx..idx+3] 的 16 字节 reinterpret 成 float4，一次读入
    float4 reg_a = FLOAT4(a[idx]);
    float4 reg_b = FLOAT4(b[idx]);
    float4 reg_c;
    // float4 有 x/y/z/w 四个分量，逐个相加
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    // 一次 16 字节写回 c[idx..idx+3]
    FLOAT4(c[idx]) = reg_c;
  } else if (idx < N) {
    // 尾巴（tail）：数组末尾不足 4 个元素时退回逐元素计算，
    // 保证任意 N 都正确且不越界。
    for (int i = 0; (idx + i) < N; i++) {
      c[idx + i] = a[idx + i] + b[idx + i];
    }
  }
}

// ============================================================================
// Kernel 3：FP16 标量版
// ============================================================================
// half 是 IEEE 754 半精度浮点，只占 2 字节（float 的一半）。
// 相同显存带宽下，FP16 理论上能搬运两倍元素，代价是精度更低
// （half 约 3~4 位十进制有效数字，保守按约 3 位）。
//
// 注意：half 加法和 float 不同，不能直接写 a[idx] + b[idx]
// （取决于编译器配置），应使用 CUDA 内建函数 __hadd。
__global__ void elementwise_add_f16_kernel(half *a, half *b, half *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    c[idx] = __hadd(a[idx], b[idx]);
}

// ============================================================================
// Kernel 4：FP16 每线程 2 个元素（half2 = 2 个 half = 4 字节）
// ============================================================================
// half2 是 CUDA 内置的 2 元素 half 向量，配合 __hadd2 一次算 2 个 half，
// 每线程一次处理 2 个连续元素。
__global__ void elementwise_add_f16x2_kernel(half *a, half *b, half *c, int N) {
  // 每线程负责以 idx 为起点的 2 个连续 half
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  // 2 个元素都有效时才走 half2 向量化分支
  if ((idx + 1) < N) {
    // 从 a[idx..idx+1] 读 4 字节为一个 half2
    half2 reg_a = HALF2(a[idx]);
    half2 reg_b = HALF2(b[idx]);
    half2 reg_c;
    // half2 只有 x、y 两个分量；__hadd 对每个分量分别做 half 加法
    reg_c.x = __hadd(reg_a.x, reg_b.x);
    reg_c.y = __hadd(reg_a.y, reg_b.y);
    HALF2(c[idx]) = reg_c;
  } else if (idx < N) {
    // 尾部：不足 2 个元素时逐元素补算
    c[idx] = __hadd(a[idx], b[idx]);
  }
}

// ============================================================================
// Kernel 5：FP16 每线程 8 个元素（用 4 次 half2 操作完成）
// ============================================================================
// 每个线程负责 8 个连续 half，把它们拆成 4 个 half2（reg_*_0..3）处理。
// 相比 f16x2，每个线程做更多工作，能摊薄索引计算等固定开销。
__global__ void elementwise_add_f16x8_kernel(half *a, half *b, half *c, int N) {
  // 每线程负责以 idx 为起点的 8 个连续 half
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  // 8 个元素都有效才走向量化主体
  if ((idx + 7) < N) {
    // 一次读入 a 侧 4 个 half2：覆盖 a[idx .. idx+7]
    half2 reg_a_0 = HALF2(a[idx + 0]);
    half2 reg_a_1 = HALF2(a[idx + 2]);
    half2 reg_a_2 = HALF2(a[idx + 4]);
    half2 reg_a_3 = HALF2(a[idx + 6]);
    // 一次读入 b 侧 4 个 half2：覆盖 b[idx .. idx+7]
    half2 reg_b_0 = HALF2(b[idx + 0]);
    half2 reg_b_1 = HALF2(b[idx + 2]);
    half2 reg_b_2 = HALF2(b[idx + 4]);
    half2 reg_b_3 = HALF2(b[idx + 6]);
    // 结果也用 4 个 half2 保存
    half2 reg_c_0, reg_c_1, reg_c_2, reg_c_3;
    // 每个 half2 的两个分量分别相加
    reg_c_0.x = __hadd(reg_a_0.x, reg_b_0.x);
    reg_c_0.y = __hadd(reg_a_0.y, reg_b_0.y);
    reg_c_1.x = __hadd(reg_a_1.x, reg_b_1.x);
    reg_c_1.y = __hadd(reg_a_1.y, reg_b_1.y);
    reg_c_2.x = __hadd(reg_a_2.x, reg_b_2.x);
    reg_c_2.y = __hadd(reg_a_2.y, reg_b_2.y);
    reg_c_3.x = __hadd(reg_a_3.x, reg_b_3.x);
    reg_c_3.y = __hadd(reg_a_3.y, reg_b_3.y);
    // 结果写回 c[idx .. idx+7]
    HALF2(c[idx + 0]) = reg_c_0;
    HALF2(c[idx + 2]) = reg_c_1;
    HALF2(c[idx + 4]) = reg_c_2;
    HALF2(c[idx + 6]) = reg_c_3;
  } else if (idx < N) {
    // 尾部：不足 8 个元素时退回逐元素计算
    for (int i = 0; (idx + i) < N; i++) {
      c[idx + i] = __hadd(a[idx + i], b[idx + i]);
    }
  }
}

// ============================================================================
// Kernel 6：FP16 每线程 8 个元素（128 位打包版）
// ============================================================================
// 8 个 half 恰好是 16 字节 = 128 位，可以把它们看成一段连续内存，
// 用 reinterpret 成 float4 的方式一次 load/store 128 位，
// 比 Kernel 5 的多次 half2 访存指令更少。
__global__ void elementwise_add_f16x8_pack_kernel(half *a, half *b, half *c,
                                                  int N) {
  // 每线程负责以 idx 为起点的 8 个连续 half
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  if ((idx + 7) < N) {
    // 局部打包数组：8 个 half 正好 128 位。
    // 这类“可寻址”局部数组在 PTX 中通常对应 .local 空间；
    // 但配合 #pragma unroll 和编译期常数下标，编译器有机会把它优化到
    // 寄存器中，从而避免真正访问 local memory。
    half pack_a[8], pack_b[8], pack_c[8]; // 8x16 bits=128 bits.
    // 把 a[idx..idx+7] 的 16 字节整体 reinterpret 成 float4，
    // 用一条访存指令读入 128 位。
    LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]); // load 128 bits
    LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]); // load 128 bits

    // 循环展开：i 每次 +2，相当于遍历 4 个 half2
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      // 每轮用 __hadd2 一次计算 2 个 half 的和，共 4 轮
      HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
    }
    // 结果 pack_c[0..7] 共 128 位，一条 store 指令整体写回 c[idx..idx+7]
    LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);
  } else if (idx < N) {
    // 尾部：不足 8 个元素时退回逐元素计算
    for (int i = 0; (idx + i) < N; i++) {
      c[idx + i] = __hadd(a[idx + i], b[idx + i]);
    }
  }
}

// ----------------------------------------------------------------------------
// 字符串化宏：把函数名转成字符串，供 PYBIND11 注册 Python 函数时使用。
// #str 会把参数按字面内容生成字符串，例如 STRINGFY(add) 得到 "add"。
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
// 展开后生成形如 elementwise_add_f32(torch::Tensor a, b, c) 的函数：
//   1. 校验三个张量的 dtype；
//   2. 根据张量形状计算 block / grid；
//   3. 用 data_ptr() 拿裸指针并启动对应 kernel。
//
// 启动策略：
// - 非二维张量：展平成 N，每个 block 固定处理 256 个元素，
//   所以 block = 256 / n_elements，grid = ceil(N / 256)；
// - 二维 (S, K)：每行需要的线程数不超过 1024 时采用“一行一个 block”
//   （grid = S, block = K / n_elements）；行过长则回退到上面的展平策略。
// ============================================================================
#define TORCH_BINDING_ELEM_ADD(packed_type, th_type, element_type, n_elements) \
  void elementwise_add_##packed_type(torch::Tensor a, torch::Tensor b,         \
                                     torch::Tensor c) {                        \
    CHECK_TORCH_TENSOR_DTYPE(a, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(b, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(c, (th_type))                                     \
    const int ndim = a.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= a.size(i);                                                        \
      }                                                                        \
      dim3 block(256 / (n_elements));                                          \
      dim3 grid((N + 256 - 1) / 256);                                          \
      elementwise_add_##packed_type##_kernel<<<grid, block>>>(                 \
          reinterpret_cast<element_type *>(a.data_ptr()),                      \
          reinterpret_cast<element_type *>(b.data_ptr()),                      \
          reinterpret_cast<element_type *>(c.data_ptr()), N);                  \
    } else {                                                                   \
      const int S = a.size(0);                                                 \
      const int K = a.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        dim3 block(K / (n_elements));                                          \
        dim3 grid(S);                                                          \
        elementwise_add_##packed_type##_kernel<<<grid, block>>>(               \
            reinterpret_cast<element_type *>(a.data_ptr()),                    \
            reinterpret_cast<element_type *>(b.data_ptr()),                    \
            reinterpret_cast<element_type *>(c.data_ptr()), N);                \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= a.size(i);                                                      \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        elementwise_add_##packed_type##_kernel<<<grid, block>>>(               \
            reinterpret_cast<element_type *>(a.data_ptr()),                    \
            reinterpret_cast<element_type *>(b.data_ptr()),                    \
            reinterpret_cast<element_type *>(c.data_ptr()), N);                \
      }                                                                        \
    }                                                                          \
  }

// 实例化 6 个 host 启动函数，分别对应 6 个 kernel：
//   f32        : float，每线程 1 元素
//   f32x4      : float，每线程 4 元素（float4）
//   f16        : half，每线程 1 元素
//   f16x2      : half，每线程 2 元素（half2）
//   f16x8      : half，每线程 8 元素（4 次 half2）
//   f16x8_pack : half，每线程 8 元素（128 位打包访存）
TORCH_BINDING_ELEM_ADD(f32, torch::kFloat32, float, 1)
TORCH_BINDING_ELEM_ADD(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_ELEM_ADD(f16, torch::kHalf, half, 1)
TORCH_BINDING_ELEM_ADD(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_ELEM_ADD(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_ELEM_ADD(f16x8_pack, torch::kHalf, half, 8)

// ============================================================================
// PyTorch C++ 扩展模块入口
//
// TORCH_EXTENSION_NAME 由 PyTorch 在编译时注入（这里实际是 elementwise_lib）。
// m.def 把上面生成的 6 个函数注册为 Python 模块里的同名函数，
// 之后 Python 里就可以直接调用 lib.elementwise_add_f32(...) 等。
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 逐个把 C++ host 函数暴露给 Python
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8_pack)
}
