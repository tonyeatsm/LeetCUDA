// ============================================================================
// histogram.cu —— CUDA 整数直方图统计（histogram）教学示例
//
// 本文件实现 y[v] = count(a[i] == v)，其中 i = 0 .. N-1。
// 输入 a 是一维 int32 数组；输出 y 是长度为 M+1 的计数数组，
// 其中 M = max(a)，桶编号范围正好是 [0, M]。
//
// 一句话理解：直方图就是「按值分堆数个数」——
// 数值相同的元素看成放进了同一个桶，最后统计每个桶里有多少个元素。
//
// 例：a = [0, 1, 2, 0, 1, 0]
//     => y[0] = 3, y[1] = 2, y[2] = 1
//
// 文件中包含 2 个计算版本：
//   1. i32   : int32 标量版（每线程 1 个元素）
//   2. i32x4 : int32 向量化版（每线程 4 个元素，16 字节访存）
//
// 学习重点（可以对照 elementwise 模块一起看）：
// - 线程编号与越界保护的写法完全相同：
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < N) ...
// - 关键差异在“输出下标由谁决定”：
//     elementwise：c[idx] = a[idx] + b[idx]，输出下标 = 线程编号，
//                  一个输出位置只被一个线程写，直接赋值即可；
//     histogram  ：y[a[idx]] += 1，输出下标 = 数据值，
//                  多个线程可能同时命中同一个桶 y[v]，
//                  必须用 atomicAdd 把“读-改-写”做成一次原子操作，
//                  否则并发累加会丢计数。
// - 向量化（int4）在这里减少的是访存指令条数；4 次 atomicAdd 之间的
//   桶冲突依然存在。热点桶的原子竞争是本模块的主要性能瓶颈，也是后续
//   用 shared memory 局部直方图 / warp 级聚合来优化的原因。
//
// 文件后半部分是 PyTorch 绑定：把 CUDA kernel 包装成 Python 可调用的函数，
// 供 histogram.py 编译与测试使用。
// ============================================================================

#include <algorithm>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <tuple>
#include <vector>

// WARP_SIZE：一个 warp（线程束）固定有 32 个线程，
// 现代 NVIDIA GPU 调度和执行指令的最小单位。
// 本文件没有直接用到它，保留是为了与其他 kernel 文件保持一致的宏集合。
#define WARP_SIZE 32

// ----------------------------------------------------------------------------
// 访存“重解释（reinterpret）”宏
//
// GPU 一次访存指令能搬移的位宽有限。把一段连续内存 reinterpret 成更宽的
// CUDA 内置向量类型，可以让编译器生成更宽的 load/store 指令。
//
// 用法示例：INT4(a[idx]) 把 &a[idx] 起的 16 字节当成一个 int4 读写。
// reinterpret_cast<T*>(&(value)) 取得该内存的 T*，[0] 取出第 1 个 T。
// 要求 value 对应地址按 16 字节（或相应类型大小）对齐，torch 分配的张量
// 通常满足要求。
//
// 说明：FLOAT4 在本文件里没有使用，保留是为了与 elementwise.cu 保持相同的
// 宏集合，方便两个模块对照阅读。
// ----------------------------------------------------------------------------
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

// ============================================================================
// Kernel 1：int32 标量版（最朴素的写法，作为正确性基准）
// ============================================================================
//
// ---- 调用链：谁触发了这个 kernel？ ----
// histogram.py
//   └─ lib.histogram_i32(a)                         # Python 调用（pybind11）
//       └─ histogram_i32(a)                         # C++ host 包装函数（宏生成，
//                                                   #  运行在 CPU 上）
//           └─ histogram_i32_kernel<<<grid, block>>>(...)  # CUDA 启动语法
//               └─ cudaLaunchKernel(...)            # nvcc 生成，交给 CUDA runtime
//                   └─ GPU 硬件并行执行本函数体
//
// 关键点：
// - histogram_i32 是 CPU 端包装函数，负责类型检查、求最大值、建桶、
//   计算 grid/block；它由宏 TORCH_BINDING_HIST 生成，直接搜索不到定义。
// - histogram_i32_kernel 是 GPU 端 __global__ 函数，
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
// ---- 为什么这里必须用 atomicAdd ----
// elementwise 中一个输出位置只被一个线程写，所以可以放心地直接赋值：
//     c[idx] = a[idx] + b[idx];
// 本 kernel 的输出位置由数据值决定：
//     int v = a[idx];   // 桶编号
//     y[v] += 1;        // 多个线程可能同时写同一个 y[v]
// 两个线程同时执行“读-改-写”时会发生竞争：
//     线程 0: read  y[v] = 0
//     线程 1: read  y[v] = 0
//     线程 0: write y[v] = 1
//     线程 1: write y[v] = 1   // 本应得到 2，实际得到 1，丢了一次计数
// atomicAdd(&y[v], 1) 把“读取 + 加 1 + 写回”交给硬件作为一条不可分割的
// 原子操作完成，从而保证并发累加的正确性。
// 这也是本模块与 elementwise 最本质的区别：
// elementwise 是“没有数据依赖的尴尬并行”，histogram 是“带写入冲突的并行”。
//
// 约定启动方式（与 elementwise 相同）：每个 block 有 256 个线程，
// 每线程负责 1 个元素，因此每个 block 处理 256 个元素，
// grid 数量取 (N + 255) / 256（向上取整）。
//
// 参数：
//   a : 输入数组（设备端指针），元素为非负整数，其值就是桶编号
//   y : 输出数组（设备端指针），长度为 M+1，存放每个桶的计数
//   N : 元素个数
__global__ void histogram_i32_kernel(int *a, int *y, int N) {
  // 全局线程编号：前面 blockIdx.x 个 block 的全部线程 + 自己在 block 内编号
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // 越界保护：N 不一定是启动线程总数的整数倍，多余线程不参与计算
  if (idx < N)
    // a[idx] 给出桶编号，把该桶的计数加 1；
    // 必须用 atomicAdd，原因见上方“为什么这里必须用 atomicAdd”
    atomicAdd(&(y[a[idx]]), 1);
}

// ============================================================================
// Kernel 2：int32 向量化版（int4 = 4 个 int = 16 字节）
// ============================================================================
// 每个线程连续处理 4 个元素，用一次 16 字节的向量 load
// 代替 4 次 4 字节访存，从而减少访存指令条数；
// 读进来之后再对 int4 的 x/y/z/w 四个分量分别做 atomicAdd。
//
// host 端对应启动：block = 256 / 4 = 64 线程，每个 block 仍处理 256 个元素。
//
// 与 elementwise 的 f32x4 对照：
// - elementwise 的 f32x4 用 (idx + 3) < N 判断整段是否越界，
//   并在尾部（tail）退回标量循环，保证任意 N 都正确；
// - 本 kernel 目前只判断了 idx < N，没有 tail 分支：
//   当 N 不是 4 的整数倍时，最后一个线程的 INT4(a[idx]) 会越界读取
//   1~3 个元素。当前测试数据长度为 10000（4 的倍数），因此没有暴露该问题。
//   本次改动只增加注释、不修改逻辑，该问题保留，供后续单独修复。
__global__ void histogram_i32x4_kernel(int *a, int *y, int N) {
  // 基础线程编号 t 对应“第 t 组 4 元素”的起点，真实下标是 4 * t
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  // 注意：这里只保证起点 idx 不越界，不保证 idx+1..idx+3 也不越界
  if (idx < N) {
    // 把 a[idx..idx+3] 的 16 字节 reinterpret 成 int4，一次读入 4 个元素
    int4 reg_a = INT4(a[idx]);
    // int4 有 x/y/z/w 四个分量，逐个做原子加 1
    atomicAdd(&(y[reg_a.x]), 1);
    atomicAdd(&(y[reg_a.y]), 1);
    atomicAdd(&(y[reg_a.z]), 1);
    atomicAdd(&(y[reg_a.w]), 1);
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

// ----------------------------------------------------------------------------
// 形状检查宏：校验张量第 0 维是否等于期望长度。
// 本文件当前的 host 函数没有用到它（输入长度由 a.size(0) 动态取得），
// 保留是为了与其他 kernel 文件保持一致的宏集合。
// ----------------------------------------------------------------------------
#define CHECK_TORCH_TENSOR_SHAPE(T, S0)                                        \
  if (((T).size(0) != (S0))) {                                                 \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

// ============================================================================
// host 启动函数生成宏
//
// 宏参数：
//   packed_type : 命名后缀，如 i32 / i32x4
//   th_type     : 对应的 torch dtype（本文件两个版本都是 torch::kInt32）
//   element_type: kernel 中的 C++ 数据类型（本文件两个版本都是 int）
//   n_elements  : 每个线程一次处理的元素个数（向量宽度，1 或 4）
//
// 展开后生成形如 histogram_i32(torch::Tensor a) 的 host 函数，
// 它依次做四件事：
//   1. 校验输入张量的 dtype 是 int32；
//   2. 用 torch::max(a, 0) 求最大值 M，再创建长度 M+1 的输出桶数组 y，
//      于是桶编号范围正好是 [0, M]，初始计数为 0；
//   3. 计算 grid / block：每个 block 固定处理 256 个元素，
//      所以 block = 256 / n_elements，grid = ceil(N / 256)；
//   4. 用 data_ptr() 取裸指针并启动对应的 kernel，最后返回 y。
//
// 注意 torch::max(a, 0) 返回的是 (values, indices) 这个 tuple，
// 这里只取 values（std::get<0>），再用 .cpu() 搬到主机端读成 int。
// 因此输出桶数由输入数据的最大值动态决定，输入必须是非负整数。
// ============================================================================
#define TORCH_BINDING_HIST(packed_type, th_type, element_type, n_elements)     \
  /* ① 宏名与 4 个参数：packed_type / th_type / element_type / n_elements */   \
  /* ② 函数签名：「造函数」——为 Python 生成 histogram_<packed_type>(a)， */    \
  /*    供 Python 侧直接调用；返回值就是统计结果张量。 */                      \
  torch::Tensor histogram_##packed_type(torch::Tensor a) {                     \
  /* ③ dtype 检查：输入必须是 th_type（本文件展开为 torch::kInt32） */         \
    CHECK_TORCH_TENSOR_DTYPE(a, (th_type))                                     \
  /* ④ 准备张量选项 auto options（下一行补全内容） */                          \
    auto options =                                                             \
  /* ⑤ 选项内容：dtype = kInt32，device = 0 号 GPU */                          \
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0);   \
  /* ⑥ N：输入元素总数，直方图要统计这 N 个元素 */                             \
    const int N = a.size(0);                                                   \
  /* ⑦ 求最大值：torch::max(a, 0) 返回 (values, indices) 二元组 */             \
    std::tuple<torch::Tensor, torch::Tensor> max_a = torch::max(a, 0);         \
  /* ⑧ 只取 values（第 0 个）并搬到 CPU 读成普通 int */                        \
    torch::Tensor max_val = std::get<0>(max_a).cpu();                          \
  /* ⑨ M：输入最大值，也就是最大的桶编号 */                                    \
    const int M = max_val.item().to<int>();                                    \
  /* ⑩ 建输出桶数组：长度 M + 1（桶编号 0..M），初值全 0 */                    \
    auto y = torch::zeros({M + 1}, options);                                   \
  /* ⑪ 每个 block 处理 256 个元素，故 block = 256 / n_elements */              \
    static const int NUM_THREADS_PER_BLOCK = 256 / (n_elements);               \
  /* ⑫ grid = ceil(N / 256)，加 255 再整除即向上取整 */                        \
    const int NUM_BLOCKS = (N + 256 - 1) / 256;                                \
  /* ⑬ block：每 block 线程数（i32 为 256，i32x4 为 64） */                    \
    dim3 block(NUM_THREADS_PER_BLOCK);                                         \
  /* ⑭ grid：本次启动创建多少个 block */                                       \
    dim3 grid(NUM_BLOCKS);                                                     \
  /* ⑮ 启动对应 kernel（名字由 packed_type 拼接），配置 <<<grid, block>>> */   \
    histogram_##packed_type##_kernel<<<grid, block>>>(                         \
  /* ⑯ 实参 1：输入张量裸指针（reinterpret_cast 成 int*） */                   \
        reinterpret_cast<element_type *>(a.data_ptr()),                        \
  /* ⑰ 实参 2、3：输出桶数组指针 y 与元素个数 N */                             \
        reinterpret_cast<element_type *>(y.data_ptr()), N);                    \
  /* ⑱ 把结果张量 y 返回给 Python 侧 */                                        \
    return y;                                                                  \
  /* ⑲ host 函数结束（宏展开到此结束） */                                      \
  }

// 实例化 2 个 host 启动函数，分别对应 2 个 kernel：
//   i32   : int，每线程 1 元素，block = 256 线程
//   i32x4 : int，每线程 4 元素（int4），block = 256 / 4 = 64 线程
TORCH_BINDING_HIST(i32, torch::kInt32, int, 1)
TORCH_BINDING_HIST(i32x4, torch::kInt32, int, 4)

// ============================================================================
// PyTorch C++ 扩展模块入口
//
// TORCH_EXTENSION_NAME 由 PyTorch 在编译时注入（这里实际是 hist_lib）。
// m.def 把上面生成的两个函数注册为 Python 模块里的同名函数，
// 之后 Python 里就可以直接调用 lib.histogram_i32(...) 与
// lib.histogram_i32x4(...)。
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 逐个把 C++ host 函数暴露给 Python
  TORCH_BINDING_COMMON_EXTENSION(histogram_i32)
  TORCH_BINDING_COMMON_EXTENSION(histogram_i32x4)
}
