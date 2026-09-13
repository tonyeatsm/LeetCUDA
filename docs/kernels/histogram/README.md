# Histogram（整数直方图统计）模块设计说明

## 模块目标

本模块演示 CUDA 中基于 **原子操作（atomic）** 的直方图统计问题：

```text
y[v] = count(a[i] == v),  i = 0 .. N-1
```

输入 `a` 是一个一维 `int32` 张量，输出 `y` 是每个整数值出现的次数。

一句话理解：直方图就是「按值分堆数个数」——数值相同的元素看成放进了同一个桶，
最后统计每个桶里有多少个元素。

例如：

```text
a = [0, 1, 2, 0, 1, 0]
=> y[0] = 3, y[1] = 2, y[2] = 1
```

和 elementwise 不同的是，多个线程可能同时命中同一个桶 `y[v]`，因此**不能直接写**
`y[a[idx]] += 1`，必须使用 `atomicAdd` 保证结果正确。

## 本次改动范围

本模块参照 elementwise 模块的做法，统一补充教学注释，改动原则是
**只加注释、不改 kernel 逻辑**，具体包含：

1. 先更新本文档（`docs/kernels/histogram/README.md`）；
2. 给 `kernels/histogram/histogram.cu` 补充分节标题、调用链、grid/block 速查、
   原子操作竞争说明与逐行行内注释；
3. 给 `kernels/histogram/histogram.py` 补充模块 docstring 与行内注释，并新增
   打印 PyTorch CUDA 扩展实际构建目录、以及打印当前 GPU 设备名的代码
   （对齐 `elementwise.py`）。

`histogram_i32x4_kernel` 缺少尾部（tail）分支的问题在本文档中已识别并说明，
但**本次不修复**，留待后续单独处理。

## 涉及文件

| 文件 | 作用 |
| --- | --- |
| `kernels/histogram/histogram.cu` | CUDA kernel 定义 + PyTorch C++ 绑定（代码中已添加详细注释） |
| `kernels/histogram/histogram.py` | 用 `torch.utils.cpp_extension.load` 编译并执行简单测试 |
| `kernels/histogram/README.md` | 模块使用说明与测试输出 |

## 注释导览

对照 elementwise 模块，`histogram.cu` 中的注释按下面的顺序组织：

| 位置 | 内容 |
| --- | --- |
| 文件头 | 模块目标（`y[v] = count(a[i] == v)`）、两个 kernel 概览、学习重点 |
| 宏定义段 | `reinterpret_cast` 向量访存宏的用法与对齐要求 |
| `histogram_i32_kernel` | 调用链、grid/block 速查、为什么必须用 `atomicAdd`、逐行注释 |
| `histogram_i32x4_kernel` | `int4` 向量化访存、与 elementwise `f32x4` 的对照、缺少 tail 分支的风险 |
| `TORCH_BINDING_HIST` | host 端宏展开细节（求最大值、建桶、计算 grid/block）与调用链 |
| `PYBIND11_MODULE` | 导出给 Python 的函数名 |

其中 `TORCH_BINDING_HIST` 采用了**宏内逐行注释**：宏体每一行代码前都配有一行
对应的 `/* ... */` 块注释，说明该行在做什么。

这里不能用 `//` 行注释：宏定义每行以 `\` 续行，而 `//` 会把该行末尾的续行符
一起注释掉，宏定义会当场断开。`/* ... */` 块注释在预处理阶段被替换为空格，
不会影响续行，因此宏内注释统一使用这种写法。

## CUDA 核心概念回顾

### 1. 线程组织

和 elementwise 一样，kernel 启动后每个线程通过以下内置变量确定自己的位置：

| 变量 | 含义 |
| --- | --- |
| `threadIdx.x` | 当前线程在 block 内的编号 |
| `blockIdx.x` | 当前 block 在 grid 内的编号 |
| `blockDim.x` | 每个 block 的线程数 |

全局线程编号为：

```c
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

### grid / block 知识点速查

- 一次 kernel 启动 `kernel<<<grid, block>>>(...)` 只创建一个 grid；
  `grid` 是“本次启动要创建的所有 block 的集合”。
- 尖括号内两个配置的含义（标准写法为 `<<<Dg, Db>>>`）：
  - `grid`：本 grid 一共有几个 block，进入 kernel 后对应 `gridDim.x`；
  - `block`：每个 block 有几个线程，进入 kernel 后对应 `blockDim.x`。
  - `grid` / `block` 只是本文件里 `dim3` 局部变量的名字，不是 CUDA 关键词。
- kernel 内部固定内置变量：
  - `gridDim.x`：总 block 数；`blockIdx.x`：当前 block 的编号（从 0 开始）；
  - `blockDim.x`：每 block 线程数；`threadIdx.x`：线程在 block 内编号（从 0 开始）。
- 全局线程编号 `idx = blockIdx.x * blockDim.x + threadIdx.x`：
  只需要知道“当前 block 前面有几个 block”，不需要知道 `gridDim.x`。
- 本模块的约定是**每个 block 处理 256 个元素**：标量版 block = 256 线程，
  向量化版 block = 256 / 4 = 64 线程（每线程 4 个元素），两者 grid 都取
  `ceil(N / 256)`。

### 2. 为什么直方图需要 atomicAdd

普通 elementwise 任务中，每个输出位置只被一个线程写，因此可以直接赋值：

```c
c[idx] = a[idx] + b[idx];
```

直方图中，输出位置由输入值决定：

```c
int v = a[idx];   // 桶编号
y[v] += 1;        // 多个线程可能同时写同一个 y[v]
```

如果两个线程同时执行 `y[v] += 1`，会发生“读旧值、各自加 1、再写回”的竞争：

```text
线程 0: read y[v] = 0
线程 1: read y[v] = 0
线程 0: write y[v] = 1
线程 1: write y[v] = 1   // 本应得到 2，实际得到 1
```

`atomicAdd(&y[v], 1)` 会把“读取 + 加 1 + 写回”作为**一个不可分割的原子操作**
交给硬件完成，从而保证并发写入正确。

### 3. 本模块的简化假设

当前实现默认输入满足：

- `a` 是一维 `int32` 张量；
- 所有元素均为非负整数；
- 直方图桶数由 `max(a) + 1` 决定。

如果输入含负数，`y[a[idx]]` 会越界，因此当前模块不处理负值输入。

## Kernel 设计

两个 kernel 都把输入输出当作一维数组处理，启动端固定让**每个 block 处理
256 个元素**。

### 1. `histogram_i32_kernel`（标量版）

```c
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N)
  atomicAdd(&y[a[idx]], 1);
```

- 一个线程处理 1 个元素；
- `if (idx < N)` 处理 `N` 不能被 block 大小整除时“多启动”的线程；
- 每个有效线程把自己的值 `a[idx]` 对应桶加 1。

### 2. `histogram_i32x4_kernel`（int4 向量化版）

每个线程连续处理 4 个 `int32`（共 16 字节）：

```c
int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
if (idx < N) {
  int4 reg_a = INT4(a[idx]);
  atomicAdd(&y[reg_a.x], 1);
  atomicAdd(&y[reg_a.y], 1);
  atomicAdd(&y[reg_a.z], 1);
  atomicAdd(&y[reg_a.w], 1);
}
```

- 将 `a[idx..idx+3]` 的 16 字节 reinterpret 成 `int4`，减少访存指令数；
- 四个分量分别做 `atomicAdd`。

### 3. 当前边界约束（已识别，本次不修复）

`histogram_i32x4_kernel` 目前**没有尾部分支**，只判断了 `idx < N`。
当 `N` 不是 4 的整数倍时，最后一个线程可能越界读取 1~3 个元素。
当前 Python 测试数据长度为 `10000`，正好是 4 的倍数，因此不会暴露该问题。

本次只加注释、不改逻辑，所以该风险仅在源码注释中写明，行为保持不变。

如需在生产中使用，应补充类似 elementwise 的 tail 分支：

```c
if ((idx + 3) < N) {
  // 4 个元素都有效，走 int4
} else {
  // 尾部不足 4 个元素，退回逐元素处理
}
```

## 启动配置（host 端）

host 端先通过 `torch::max(a, 0)` 得到输入最大值 `M`，再创建长度 `M + 1`
的 `int32` 输出张量，桶编号范围是 `[0, M]`：

```c
const int M = max_val.item().to<int>();
auto y = torch::zeros({M + 1}, options);
```

随后按每个 block 处理 256 个元素计算启动配置：

```c
static const int NUM_THREADS_PER_BLOCK = 256 / n_elements;
const int NUM_BLOCKS = (N + 256 - 1) / 256;
dim3 block(NUM_THREADS_PER_BLOCK);
dim3 grid(NUM_BLOCKS);
```

所以：

| 版本 | 每线程元素数 | block 线程数 | 每 block 处理元素数 |
| --- | ---: | ---: | ---: |
| `histogram_i32` | 1 | 256 | 256 |
| `histogram_i32x4` | 4 | 64 | 256 |

## PyTorch 绑定

`histogram.cu` 通过 `TORCH_BINDING_HIST` 宏生成两个 host 函数，再经
`PYBIND11_MODULE` 暴露给 Python：

```text
histogram.py
  └─ lib.histogram_i32(a)                        # Python 调用（pybind11）
      └─ histogram_i32(torch::Tensor a)          # C++ host 包装函数（宏生成）
          ├─ torch::max(a, 0)                    # 求最大值，确定桶数量
          ├─ torch::zeros({M + 1}, int32 CUDA)   # 创建输出桶数组
          └─ histogram_i32_kernel<<<grid, block>>>(...)  # 启动 GPU kernel
              └─ GPU 各线程 atomicAdd 到对应桶
```

host 函数的主要工作：

1. `CHECK_TORCH_TENSOR_DTYPE` 检查输入为 `torch::kInt32`；
2. 用 `torch::max` 得到最大值 `M`；
3. 创建 `M + 1` 个桶；
4. 计算 `grid / block`；
5. 取裸指针并启动对应 kernel。

## 测试脚本

`histogram.py` 流程：

1. 用 `torch.utils.cpp_extension.load` 现场编译 `histogram.cu`；
2. 用 `torch.utils.cpp_extension._get_build_directory("hist_lib", False)` 打印
   PyTorch CUDA 扩展的实际构建目录（`hist_lib.so` 所在位置），
   再用 `torch.cuda.get_device_name()` 打印当前 GPU 设备名；
3. 生成 `a = [0,1,...,9] * 1000`，长度 `10000`；
4. 分别调用 `histogram_i32` 和 `histogram_i32x4`；
5. 打印每个桶的计数值。

预期每个值出现 `1000` 次：

```text
h_i32   0: 1000
h_i32   1: 1000
...
h_i32   9: 1000
h_i32x4 0: 1000
h_i32x4 1: 1000
...
h_i32x4 9: 1000
```

## 后续可以尝试的优化方向

- 补上 `histogram_i32x4_kernel` 的尾部分支，使任意 `N` 都安全；
- 使用 shared memory 做 block 内局部直方图，最后再合并到全局，减少原子冲突；
- 使用 warp 内 `__ballot_sync` / `__match_any_sync` 等归约手段降低竞争；
- 支持更大桶数或负值索引；
- 对比不同 block 大小、不同向量宽度对直方图吞吐的实测影响。
