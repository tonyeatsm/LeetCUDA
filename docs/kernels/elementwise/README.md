# Elementwise（逐元素加法）模块设计说明

## 模块目标

本模块用于演示 CUDA 中最基础的 **elementwise（逐元素）** 运算：

```text
c[i] = a[i] + b[i],  i = 0 .. N-1
```

同时展示同一个运算的不同优化写法：

1. 标量版本：一个线程处理一个元素；
2. 向量化版本（`float4` / `half2` / 128-bit 打包）：一个线程一次处理多个连续元素；
3. FP16 版本：用更小精度的数据类型，成倍降低访存字节数。

最终目标不是追求极致性能，而是让学习者直观理解 CUDA 编程模型与访存优化思想。

## 涉及文件

| 文件 | 作用 |
| --- | --- |
| `kernels/elementwise/elementwise.cu` | CUDA kernel 定义 + PyTorch C++ 绑定（代码中已添加详细注释） |
| `kernels/elementwise/elementwise.py` | 用 `torch.utils.cpp_extension.load` 编译并执行基准测试 |
| `kernels/elementwise/README.md` | 模块使用说明与测试输出 |

## CUDA 核心概念回顾

### 1. 线程组织

启动一个 kernel 时，GPU 会按两层结构创建线程：

```text
grid（若干 block）
└── block（若干 thread）
    └── thread
```

每个线程用内置变量确定自己的位置：

| 变量 | 含义 |
| --- | --- |
| `threadIdx.x` | 当前线程在 block 内的编号 |
| `blockIdx.x` | 当前 block 在 grid 内的编号 |
| `blockDim.x` | 每个 block 的线程数 |

因此，线程的全局编号为：

```c
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

### grid / block 知识点速查

- 一次 kernel 启动 `kernel<<<grid, block>>>(...)` 只创建一个 grid；
  `grid` 是“本次启动要创建的所有 block 的集合”。
- 尖括号内两个配置的含义（标准写法为 `<<<Dg, Db>>>`）：
  - `grid`：本 grid 一共有几个 block，进入 kernel 后对应 `gridDim.x`；
  - `block`：每个 block 有几个线程，进入 kernel 后对应 `blockDim.x`。
  - `grid` / `block` 只是本文件里 `dim3` 局部变量的名字，不是 CUDA 关键词，
    也可以写成 `<<<blocksPerGrid, threadsPerBlock>>>`。
- kernel 内部固定内置变量：
  - `gridDim.x`：总 block 数；`blockIdx.x`：当前 block 的编号（从 0 开始，
    在本次启动的 grid 内唯一）；
  - `blockDim.x`：每 block 线程数；`threadIdx.x`：线程在 block 内编号（从 0 开始）。
- 全局线程编号 `idx = blockIdx.x * blockDim.x + threadIdx.x`：
  只需要知道“当前 block 前面有几个 block”，不需要知道 `gridDim.x`。
- 例：`<<<4, 256>>>`、`N = 1000`：
  - `gridDim.x = 4`、`blockDim.x = 256`，总线程数 1024；
  - block 0/1/2/3 分别负责下标 0~255、256~511、512~767、768~1023；
  - 下标 1000~1023 的 24 个线程被 `if (idx < N)` 拦截，不参与计算。
- 同一 kernel 函数可以被多次启动，每次启动产生一个独立 grid；
  因此准确说法是“一次启动 = 一个 grid”，而不是“一个 kernel = 一个 grid”。

### 2. 为什么 elementwise 任务适合 GPU

`c[i] = a[i] + b[i]` 中，每个输出只依赖对应位置的输入，线程之间**没有依赖、无需通信**，
属于"尴尬并行"（embarrassingly parallel）问题，可以放心地把数组切成很多份并行处理。

### 3. 优化这类任务的关键

GPU 加法本身极快，瓶颈往往在**显存带宽**。常见优化手段：

- **访存合并（coalescing）**：让同一时刻相邻的线程访问相邻地址，使一次访存事务能服务整批线程；
- **向量化**：每个线程一次读写 4/8/16 字节甚至 128 位，减少访存指令条数；
- **降低数据类型宽度**：FP16 只有 FP32 一半字节数，同样带宽下能传输两倍元素。

## Kernel 设计

所有 kernel 都遵守一个约定：**把输入输出都当作长度为 `N` 的一维数组处理**。
二维矩阵 `(S, K)` 在启动端被展平为 `N = S * K`。

### 1. `elementwise_add_f32_kernel`（FP32 标量版）

```c
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N) c[idx] = a[idx] + b[idx];
```

- 一个线程只算一个元素；
- `if (idx < N)` 处理 `N` 不能被 block 大小整除时"多启动"的线程；
- 正确性最直观，是后面所有版本的基准。

### 2. `elementwise_add_f32x4_kernel`（FP32 向量化版）

每个线程连续处理 **4 个 float（16 字节）**：

```c
int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
```

- 将一段连续内存 reinterpret 成 `float4`，用一次加载/存储完成 16 字节搬运；
- 必须用 `(idx + 3) < N` 判断整段是否越界；
- 尾部不足 4 个的元素用标量循环补算，保证任何 `N` 都正确。

### 3. `elementwise_add_f16_kernel`（FP16 标量版）

- 数据类型为 `half`（2 字节），加法使用 CUDA 内建函数 `__hadd`；
- 字节数减半，是后续 FP16 向量化的基础。
- 精度代价：FP32 尾数 23 位，最小刻度约 `2^-23 ≈ 0.00000012`（约 7 位有效数字）；
  FP16 尾数只有 10 位，最小刻度为
  `2^-10` = 2 的 10 次方分之一 = `1/1024` ≈ `0.00098`
  （约 3~4 位有效数字，保守按约 3 位）；
  FP32 转 half 会就近取整产生量化误差，例如同一数值
  FP32 显示 `1.223737`、FP16 显示 `1.22460938`，属于预期精度损失。

### 4. `elementwise_add_f16x2_kernel`（FP16 每线程 2 元素）

- 一次用 `half2`（2 个 half = 4 字节）完成两个元素的搬运；
- 加法用 `__hadd2`，一次算两个 half。

### 5. `elementwise_add_f16x8_kernel`（FP16 每线程 8 元素）

- 每个线程处理 8 个 half，通过 4 次 `half2` 操作完成；
- 相比 f16x2，进一步摊薄索引计算与循环开销。

### 6. `elementwise_add_f16x8_pack_kernel`（128-bit 打包版）

- 把 8 个 half（正好 16 字节）装入局部数组；
- 借助 reinterpret 成 `float4` 的 `LDST128BITS` 宏，让一次访存指令读写完整的 128 位；
- 配合 `#pragma unroll` 与编译期常数下标，编译器有机会把数组优化到寄存器，
  实际表现通常为六种版本中最快（见基准测试）。

## 启动配置（host 端）

启动端把每个 block 负责的元素数固定为 **256**：

```c
dim3 block(256 / n_elements);   // 每个线程负责 n_elements 个元素
dim3 grid((N + 256 - 1) / 256); // 向上取整
```

二维输入 `(S, K)` 且每行元素数除以 `n_elements` 不超过 1024 时，
采用"一行一个 block"的启动方式：

```c
dim3 block(K / n_elements);
dim3 grid(S);
```

当某行需要的线程数超过硬件上限（通常 1024）时，回退到展平启动。

## PyTorch 绑定

`elementwise.cu` 通过宏批量生成 6 个 host 函数，再经
`PYBIND11_MODULE` 暴露为 Python 可调用模块，主要工作：

1. 用 `CHECK_TORCH_TENSOR_DTYPE` 检查张量数据类型；
2. 根据维度/形状计算 `grid` 与 `block`；
3. 用 `reinterpret_cast<element_type *>(a.data_ptr())` 拿到裸指针；
4. 以 `kernel<<<grid, block>>>` 启动内核。

### 调用链（以 `elementwise_add_f32` 为例）

```text
elementwise.py
  └─ lib.elementwise_add_f32(a, b, c)             # Python 调用（pybind11）
      └─ elementwise_add_f32(...)                 # C++ host 包装函数（宏生成）
          └─ elementwise_add_f32_kernel<<<grid, block>>>(...)  # CUDA 启动语法
              └─ cudaLaunchKernel(...)            # nvcc 生成，交给 CUDA runtime
                  └─ GPU 硬件并行执行 kernel 函数体
```

要点：

- Python 调用的 `elementwise_add_f32` 运行在 CPU 上，负责类型检查、计算 `grid/block`；
- `elementwise_add_f32_kernel` 运行在 GPU 上，必须用 `<<<>>>` 启动，不能普通函数调用；
- 启动后 GPU 的每个线程都会执行一遍 kernel 函数体，靠 `threadIdx` / `blockIdx`
  区分自己负责的下标 `idx`。

## 基准测试脚本

`elementwise.py` 流程：

1. 用 `torch.utils.cpp_extension.load` 现场编译 `elementwise.cu`；
2. 对 `S ∈ {1024, 2048, 4096}`、`K ∈ {1024, 2048, 4096}` 组合生成随机张量；
3. `run_benchmark` 先做 warmup，再运行 1000 次取平均；
4. 依次对比 6 个自定义 kernel 与 PyTorch 官方 `torch.add` 的正确性和耗时。

脚本在 `load(...)` 完成后会用
`print(ext._get_build_directory("elementwise_lib", False))` 打印
PyTorch CUDA 扩展的实际构建目录，便于确认 `elementwise_lib.so` 的存放位置；
随后用 `print(torch.cuda.get_device_name())` 打印当前 GPU 设备名，
便于对照不同显卡上的耗时数据。

打印对照结果前，`run_benchmark` 会通过
`out.flatten().detach().cpu().numpy().tolist()[:2]` 把 GPU 上的结果张量
拉平、断开可能的梯度追踪（本脚本已关闭 autograd，属于保险写法）、拷回 CPU、
转成 Python 列表，并只取前 2 个元素用于人工核对。

## 性能观察与结论

从 `kernels/elementwise/README.md` 的测试输出可总结：

- FP16 各版本一般快于对应 FP32 版本，因为访存字节减半；
- 同数据类型下，向量化版本普遍快于标量版本；
- `f16x8_pack` 在多数规模下最快，验证"更宽的访存 + 更小的数据类型"的组合收益；
- 小矩阵或极端规模下耗时差异可能不明显，甚至出现波动，因为 kernel 时间已接近启动开销量级。

## 后续可以尝试的优化方向

- 引入 grid-stride loop，使固定大小的 grid 也能处理任意大 `N`；
- 使用 `__ldg` / 只读缓存提示；
- 在 FP16 输入转 FP32 计算再转回，提升精度（split-k 思路的简化版）；
- 对比不同 `block` 大小、不同向量宽度对带宽的实测影响。
