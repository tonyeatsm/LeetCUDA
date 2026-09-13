# ReLU（Rectified Linear Unit，修正线性单元）模块设计说明

## 模块目标

本模块用于演示 CUDA 中逐元素 **ReLU（修正线性单元）** 的实现：

```text
y[i] = max(0, x[i]),  i = 0 .. N-1
```

同时展示同一个运算的不同优化写法：

1. 标量版本：一个线程处理一个元素；
2. 向量化版本（`float4` / `half2` / 128-bit 打包）：一个线程一次处理多个连续元素；
3. FP16 版本：用更小精度的数据类型，成倍降低访存字节数。

本模块与 elementwise、sigmoid 两个模块是**一对姊妹示例**：6 个 kernel 版本一一
对应，区别只是把 `c[i] = a[i] + b[i]` / `y[i] = sigmoid(x[i])` 换成
`y[i] = max(0, x[i])`。前面模块讲过的线程编号、向量化访存、FP16 打包等知识
本模块不再重复。

与 sigmoid 相比，ReLU 是**计算极轻**的算子：一次取最大值即可完成，没有指数、
没有除法、也不会溢出。因此本模块除了教学，还承担一个"**性能上限参照**"的角色：
当计算量小到几乎可以忽略时，kernel 时间基本由访存主导，此时各种访存写法
（标量 / 向量化 / FP16 / 128 位打包）的差距会表现得最干净。

最终目标不是追求极致性能，而是让学习者直观理解 CUDA 编程模型与访存优化思想。

## 本次改动范围

本模块参照 elementwise / sigmoid 模块的做法，统一补充教学注释，改动原则是
**只加注释、不改 kernel 逻辑**，具体包含：

1. 先更新本文档（`docs/kernels/relu/README.md`）；
2. 给 `kernels/relu/relu.cu` 补充分节标题、调用链、grid/block 速查、
   ReLU 数学性质说明，以及逐行行内注释；
3. 给 `kernels/relu/relu.py` 补充模块 docstring 与行内注释，并新增
   打印 PyTorch CUDA 扩展实际构建目录、以及打印当前 GPU 设备名的代码
   （对齐 `sigmoid.py`）。

`relu.py` 的**运行行为保持不变**：官方对照仍直接调用 `torch.relu`
（不像 `sigmoid.py` 那样用 `partial(torch.sigmoid, out=y)`），结果打印仍保留
`f"{v:<12}"` 的左对齐补齐格式，因此 `kernels/relu/README.md` 中的测试输出
样例无需改动。

`scripts/README.md` 中补充了 relu 模块的运行命令。

向量化版本（`f32x4` / `f16x2` / `f16x8` / `f16x8_pack`）缺少尾部（tail）分支的
问题，以及 `fmaxf` / `__hmax` 的 NaN 语义与 `torch.relu` 不一致的问题，
在本文档中已识别并说明，但**本次不修复**，留待后续单独处理。

## 涉及文件

| 文件 | 作用 |
| --- | --- |
| `kernels/relu/relu.cu` | CUDA kernel 定义 + PyTorch C++ 绑定（代码中已添加详细注释） |
| `kernels/relu/relu.py` | 用 `torch.utils.cpp_extension.load` 编译并执行基准测试 |
| `kernels/relu/README.md` | 模块使用说明与测试输出 |

## 注释导览

对照 elementwise / sigmoid 模块，`relu.cu` 中的注释按下面的顺序组织：

| 位置 | 内容 |
| --- | --- |
| 文件头 | 模块公式、6 个版本概览、与 elementwise / sigmoid 的对应关系、学习重点 |
| 【ReLU 曲线速览】 | ASCII 折线图、采样值、分段线性 / 不可导 / 梯度不衰减 / 死亡 ReLU |
| 宏定义段 | `reinterpret_cast` 向量访存宏的用法与对齐要求、为什么不需要溢出保护 |
| `relu_f32_kernel` | 调用链、grid/block 速查、`fmaxf` 选型与 NaN 语义、逐行注释 |
| `relu_f32x4_kernel` | `float4` 向量化访存、逐分量展开计算、缺少 tail 分支的风险 |
| `relu_f16_kernel` | FP16 常数转换的开销、`__hmax` 选型 |
| `relu_f16x2_kernel` | `half2` 双分量计算、多读一次 `y` 的冗余访存 |
| `relu_f16x8_kernel` | unpack 写法：4 个 `half2` 完成 8 个元素 |
| `relu_f16x8_pack_kernel` | pack 写法：局部数组 + 128 位整体访存 + `__hmax2` 成对指令 |
| `TORCH_BINDING_RELU` | host 端宏展开细节与调用链 |
| `PYBIND11_MODULE` | 导出给 Python 的函数名 |

`TORCH_BINDING_RELU` 宏内部的注释使用 `/* ... */` 块注释而非 `//`：
宏定义每行以 `\` 续行，而 `//` 会把该行末尾的续行符一起注释掉，宏定义会当场断开。
`/* ... */` 块注释在预处理阶段被替换为空格，不会影响续行。

## CUDA 核心概念回顾

### 1. 线程组织

和 elementwise / sigmoid 一样，kernel 启动后每个线程通过以下内置变量确定自己的位置：

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
  `grid` 是"本次启动要创建的所有 block 的集合"。
- 尖括号内两个配置的含义（标准写法为 `<<<Dg, Db>>>`）：
  - `grid`：本 grid 一共有几个 block，进入 kernel 后对应 `gridDim.x`；
  - `block`：每个 block 有几个线程，进入 kernel 后对应 `blockDim.x`。
  - `grid` / `block` 只是本文件里 `dim3` 局部变量的名字，不是 CUDA 关键词。
- kernel 内部固定内置变量：
  - `gridDim.x`：总 block 数；`blockIdx.x`：当前 block 的编号（从 0 开始）；
  - `blockDim.x`：每 block 线程数；`threadIdx.x`：线程在 block 内编号（从 0 开始）。
- 全局线程编号 `idx = blockIdx.x * blockDim.x + threadIdx.x`：
  只需要知道"当前 block 前面有几个 block"，不需要知道 `gridDim.x`。
- 本模块的约定是**每个 block 处理 256 个元素**：标量版 block = 256 线程，
  向量化版 block = 256 / n_elements（每线程处理 n_elements 个元素），
  grid 都取 `ceil(N / 256)`。
- 例：`<<<4, 256>>>`、`N = 1000`：
  - `gridDim.x = 4`、`blockDim.x = 256`，总线程数 1024；
  - block 0/1/2/3 分别负责下标 0~255、256~511、512~767、768~1023；
  - 下标 1000~1023 的 24 个线程被 `if (idx < N)` 拦截，不参与计算。

### 2. ReLU 的数学性质

```text
relu(x) = max(0, x) = { x,  x > 0
                      { 0,  x <= 0
```

函数曲线（ASCII 示意图，`x ∈ [-3, 3]`，`+` 标出折点 `(0, 0)`）：

```text
 3.0 ┤                          ***
     ┤                       ***
 2.0 ┤                    ***
     ┤                 ***
 1.0 ┤              ***
     ┤           ***
 0.0 ┤***********+
     └──────────────────────────────→ x
     -3     -2    -1    0    1    2    3
```

采样值：`relu(-3) = 0`、`relu(-1) = 0`、`relu(0) = 0`、`relu(1) = 1`、
`relu(2) = 2`、`relu(3) = 3`（正半轴是恒等映射，负半轴被"整流"成 0，
这也是 Rectified 这个词的来历）。

- **分段线性、不是饱和函数**：值域是 `[0, +∞)`。sigmoid 在两端导数趋近 0，
  而 ReLU 在正半轴的导数**恒为 1**，梯度不会随 `x` 增大而衰减，
  这是它比 sigmoid 更不容易梯度消失、成为现代网络隐藏层默认选择的直接原因；
- **在 `x = 0` 处不可导**：左导数为 0、右导数为 1，工程上取次梯度
  （常用 0，PyTorch 的 `relu` 在 0 点也取 0）；
- **负半轴导数为 0**：如果一个神经元的输出长期落在负半轴，它的梯度恒为 0、
  权重再也不更新，这就是著名的 **死亡 ReLU（dead ReLU）** 问题，
  也是 leaky ReLU、GELU、SiLU 等变体出现的动机；
- **计算量极低**：整个算子只有一次比较/取最大值，没有 `exp`、没有除法，
  因此它是"访存瓶颈"最典型、最容易观察带宽上限的逐元素算子。

### 3. 为什么 ReLU 不需要溢出保护（与 sigmoid 的关键差异）

sigmoid 必须先 `clamp` 再 `exp`，否则 `exp(-x)` 会溢出（见
`docs/kernels/sigmoid/README.md` 中的 `MAX_EXP_F32` / `MAX_EXP_F16` 说明）。
ReLU 完全没有这个问题：

| 输入情况 | `relu_f32_kernel` 的结果 |
| --- | --- |
| `x` 是很大的正数（如 `3.4e38`） | 原样返回，不会溢出（不产生新的数值） |
| `x` 是很大的负数（如 `-3.4e38`） | 截断为 `0.0f` |
| `x` 是 `+inf` / `-inf` | `+inf` / `0.0f` |

只有一次比较和一次赋值，结果一定落在 `[0, +∞)` 内，因此 kernel 里既没有
`fminf` / `fmaxf` 的"夹取"步骤，也没有 `MAX_EXP_*` 这类边界宏。
唯一的坑是 **NaN 语义**，见下文"边界约束"。

### 4. FP16 精度提醒

FP32 尾数 23 位，最小刻度约 `2^-23 ≈ 0.00000012`（约 7 位有效数字）；
FP16 尾数只有 10 位，最小刻度为 `2^-10` = 2 的 10 次方分之一 = `1/1024`
≈ `0.00098`（约 3~4 位有效数字，保守按约 3 位）。

FP32 转 half 会就近取整产生量化误差，例如同一数值 FP32 显示 `0.27619576`、
FP16 显示 `0.27612305`，属于预期精度损失（`kernels/relu/README.md` 的
测试输出里可以看到这类差异）。

注意 ReLU 与 sigmoid 的差别：ReLU 是"原样返回或清零"，**误差不会被放大**，
量化误差只来自输入张量本身从 FP32 转成 FP16 的那一步；而且 FP16 的最大规格数
只有 65504，`torch.randn` 生成的输入远小于它，不会触发上溢。

## Kernel 设计

所有 kernel 都遵守一个约定：**把输入输出都当作长度为 `N` 的一维数组处理**。
二维矩阵 `(S, K)` 在启动端被展平为 `N = S * K`。

### 1. `relu_f32_kernel`（FP32 标量版）

```c
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N)
  y[idx] = fmaxf(0.0f, x[idx]);
```

- 一个线程只算一个元素；
- `if (idx < N)` 处理 `N` 不能被 block 大小整除时"多启动"的线程；
- `fmaxf` 是 float 版本的取最大值函数，不能写成 `fmax`（那是 double 版本，
  会把参数提升成 double 再比较，明显更慢）；
- 参数写成 `fmaxf(0.0f, x[idx])` 只是习惯写法，`fmaxf` 本身是对称的
  （NaN 的情况除外，见"边界约束"）；
- 正确性最直观，是后面所有版本的基准。

### 2. `relu_f32x4_kernel`（FP32 向量化版）

每个线程连续处理 **4 个 float（16 字节）**：

```c
int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
float4 reg_x = FLOAT4(x[idx]);
float4 reg_y;
reg_y.x = fmaxf(0.0f, reg_x.x);
...
FLOAT4(y[idx]) = reg_y;
```

- 将一段连续内存 reinterpret 成 `float4`，用一次加载/存储完成 16 字节搬运，
  4 次 4 字节访存被压缩成 1 次 16 字节访存；
- 与 elementwise 不同，ReLU 和 sigmoid 一样是**单输入**算子：只需读 `x`
  一份数据，没有 `b` 侧的第二路 load；
- ReLU 计算极轻，向量化省下的访存指令条数几乎直接体现在耗时上，
  这也是它和 sigmoid 对照观察时最直观的地方。

### 3. `relu_f16_kernel`（FP16 标量版）

```c
if (idx < N)
  y[idx] = __hmax(__float2half(0.0f), x[idx]);
```

- 数据类型为 `half`（2 字节），字节数减半，是后续 FP16 向量化的基础；
- sigmoid 版本要预先算好 `const half f = __float2half(1.0f)` 并复用，
  因为 `f` 在分子分母里各出现一次；ReLU 只需要一个常数 0，
  且 `__float2half(0.0f)` 的参数是字面量，编译器会直接折叠成 half 常量，
  不产生运行时的转换开销；
- 取最大值必须用 half 版本的 `__hmax`（直接比较 16 位 half），
  写成 `fmaxf` 会先把 half 提升到 float、算完再转回来，多出两条转换指令。

### 4. `relu_f16x2_kernel`（FP16 每线程 2 元素）

- 一次用 `half2`（2 个 half = 4 字节）完成两个元素的搬运；
- `half2` 只有 `x`、`y` 两个分量，两个分量各自独立调用 `__hmax`；
- 代码里 `half2 reg_y = HALF2(y[idx]);` 先读了一次 `y`，但紧接着
  `reg_y.x` / `reg_y.y` 都会被覆盖，这次读取对结果没有影响，
  属于冗余访存（保留原样，本次不改逻辑）。

### 5. `relu_f16x8_kernel`（FP16 每线程 8 元素，unpack 写法）

```c
int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
half2 reg_x_0 = HALF2(x[idx + 0]);
half2 reg_x_1 = HALF2(x[idx + 2]);
half2 reg_x_2 = HALF2(x[idx + 4]);
half2 reg_x_3 = HALF2(x[idx + 6]);
```

- 每个线程处理 8 个 half，拆成 4 个 `half2`（`idx+0/2/4/6`）分别处理；
- 这里的 "unpack" 指：数据在内存里本就是连续的 8 个 half，
  本 kernel 不把它整体打包搬运，而是按 `half2` 逐段读、逐段写；
- 相比 f16x2，每个线程做更多工作，能摊薄索引计算等固定开销；
- 结果是 4 个 `half2`，分别写回 `y[idx+0/2/4/6]`。

### 6. `relu_f16x8_pack_kernel`（128-bit 打包版）

```c
const half2 z2 = {__float2half(0.0f), __float2half(0.0f)};
half pack_x[8], pack_y[8]; // 8x16 bits=128 bits.
LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); // load 128 bits
#pragma unroll
for (int i = 0; i < 8; i += 2) {
  HALF2(pack_y[i]) = __hmax2(HALF2(pack_x[i]), z2);
}
LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
```

- 把 8 个 half（正好 16 字节 = 128 位）装入局部数组；
- 借助 reinterpret 成 `float4` 的 `LDST128BITS` 宏，让一次访存指令读写完整的
  128 位，比 unpack 版本的多次 `half2` 访存更省指令；
- 局部数组在 PTX 中通常对应 `.local` 空间（可寻址），
  但配合 `#pragma unroll` 与编译期常数下标，编译器有机会把它优化到寄存器里，
  从而避免真正访问 local memory；
- **这里是本模块与 sigmoid 在写法上最有意思的差异**：
  sigmoid 没有成对的 `hexp2`，循环只能 `i++` 逐个 half 调用 `hexp`；
  而 ReLU 有 `__hmax2`，一次就能处理一个 `half2`（2 个 half），
  所以循环步长是 `i += 2`，8 个元素只需 4 次成对指令。

## 边界约束（已识别，本次不修复）

elementwise 的向量化版本用 `(idx + 3) < N` 判断整段元素是否齐全，
并配了尾部标量回退分支；本模块的向量化版本**两个都没有**，
越界判断只检查了段的起始下标：

| kernel | 现有判断 | `N` 不是向量宽度的整数倍时会发生什么 |
| --- | --- | --- |
| `relu_f32x4_kernel` | `(idx + 0) < N` | 最后一个 block 的部分线程会把 `idx+1..idx+3` 写到 `y` 合法范围之外（越界写 1~3 个 float） |
| `relu_f16x2_kernel` | `(idx + 0) < N` | 同上，越界写 1 个 half |
| `relu_f16x8_kernel` | `(idx + 0/2/4/6) < N` | 同上，越界写 1 个 half |
| `relu_f16x8_pack_kernel` | `(idx + 7) < N` | 方向相反：末尾不足 8 个的元素被整块**丢弃**（漏算），但不会越界 |

本模块基准测试的 `S`/`K` 都取 1024 的倍数（元素数是 256 的倍数），
启动的线程恰好整段对齐，因此不会暴露该问题。

本次只加注释、不改逻辑，所以该风险仅在本文档和源码注释中写明。
如需在生产中使用，应参照 elementwise 补充尾部处理：

```c
if ((idx + 3) < N) {
  // 4 个元素都有效，走 float4 向量化分支
} else if (idx < N) {
  // 尾部不足 4 个元素，退回逐元素处理
}
```

### NaN 语义差异（已识别，本次不修复）

本模块 kernel 用 `fmaxf` / `__hmax` 取最大值，它们遵循 IEEE 754-2008 的
`maxNum` 语义：**一个参数是 NaN 时返回另一个参数**。因此
`relu_f32_kernel` 遇到 `x = NaN` 会输出 `0.0f`
（可用 `fmaxf(nan, 0.0f) == 0.0f` 验证），而 PyTorch 的 `torch.relu`
会把 NaN 传播到输出。两者在含 NaN 的输入上结果不同。

本模块的基准测试用 `torch.randn` 生成有限值输入，不会触发该差异；
按"只加注释、不改逻辑"的原则，本次不修改 kernel。

## 启动配置（host 端）

启动端把每个 block 负责的元素数固定为 **256**：

```c
dim3 block(256 / (n_elements));   // 每个线程负责 n_elements 个元素
dim3 grid((N + 256 - 1) / 256);   // 向上取整
```

各版本的参数对应关系：

| 版本 | `n_elements` | block 线程数 | 每 block 处理元素数 |
| --- | ---: | ---: | ---: |
| `relu_f32` | 1 | 256 | 256 |
| `relu_f32x4` | 4 | 64 | 256 |
| `relu_f16` | 1 | 256 | 256 |
| `relu_f16x2` | 2 | 128 | 256 |
| `relu_f16x8` | 8 | 32 | 256 |
| `relu_f16x8_pack` | 8 | 32 | 256 |

二维输入 `(S, K)` 且每行元素数除以 `n_elements` 不超过 1024 时，
采用"一行一个 block"的启动方式：

```c
dim3 block(K / (n_elements));
dim3 grid(S);
```

当某行需要的线程数超过硬件上限（通常 1024）时，回退到展平启动。
注意两个分支里的 `block` 都同时受 256 与硬件线程上限约束，因此
`f32x4` 在"一行一个 block"分支下的 block 大小是 `K / 4`。

## PyTorch 绑定

`relu.cu` 通过 `TORCH_BINDING_RELU` 宏批量生成 6 个 host 函数，
再经 `PYBIND11_MODULE` 暴露为 Python 可调用模块，主要工作：

1. 用 `CHECK_TORCH_TENSOR_DTYPE` 检查输入输出张量数据类型；
2. 根据维度/形状计算 `grid` 与 `block`；
3. 用 `reinterpret_cast<element_type *>(x.data_ptr())` 拿到裸指针；
4. 以 `kernel<<<grid, block>>>` 启动内核。

### 调用链（以 `relu_f32` 为例）

```text
relu.py
  └─ lib.relu_f32(x, y)                       # Python 调用（pybind11）
      └─ relu_f32(torch::Tensor x, y)         # C++ host 包装函数（宏生成）
          ├─ CHECK_TORCH_TENSOR_DTYPE(x/y)    # 校验 dtype 必须是 float32
          ├─ 计算 dim3 block / dim3 grid
          └─ relu_f32_kernel<<<grid, block>>>(...)     # CUDA 启动语法
              └─ cudaLaunchKernel(...)        # nvcc 生成，交给 CUDA runtime
                  └─ GPU 各线程并行执行 kernel 函数体
```

要点：

- Python 调用的 `relu_f32` 运行在 CPU 上，负责类型检查、计算 `grid` / `block`；
- `relu_f32_kernel` 运行在 GPU 上，必须用 `<<<>>>` 启动，不能普通函数调用；
- 启动后 GPU 的每个线程都会执行一遍 kernel 函数体，靠 `threadIdx` / `blockIdx`
  区分自己负责的下标 `idx`。

## 测试脚本

`relu.py` 流程：

1. 用 `torch.utils.cpp_extension.load` 现场编译 `relu.cu`；
2. 对 `S ∈ {1024, 2048, 4096}`、`K ∈ {1024, 2048, 4096}` 组合生成随机张量；
3. `run_benchmark` 先做 warmup，再运行 1000 次取平均；
4. 依次对比 6 个自定义 kernel 与 PyTorch 官方 `torch.relu` 的正确性和耗时。

脚本在 `load(...)` 完成后会打印 PyTorch CUDA 扩展的实际构建目录，
便于确认 `relu_lib.so` 的存放位置；随后用
`print(torch.cuda.get_device_name())` 打印当前 GPU 设备名，
便于对照不同显卡上的耗时数据。

打印对照结果前，`run_benchmark` 会通过
`out.flatten().detach().cpu().numpy().tolist()[:2]` 把 GPU 上的结果张量
拉平、断开可能的梯度追踪（本脚本已关闭 autograd，属于保险写法）、拷回 CPU、
转成 Python 列表，并只取前 2 个元素用于人工核对；本模块还额外用
`f"{v:<12}"` 把每个数值左对齐补齐到 12 个字符，所以
`kernels/relu/README.md` 里的输出带引号和多余空格，属正常格式。

另外，`torch.randn` 生成的随机张量前两个元素可能是负数，被 ReLU 截成 0，
所以测试输出里出现 `'0.0'`（如 `S=1024, K=1024` 那组）是正常现象，
不代表 kernel 出错——判断正确性要看同组里 6 个版本与 `out_f32_th`
是否**一致**。

## 性能观察与结论

从 `kernels/relu/README.md` 的测试输出可总结：

- 同一份数据下 FP16 各版本普遍明显快于 FP32 版本，因为访存字节数减半，
  例如 `S=4096, K=4096` 时 `f32` 约 0.1885 ms、`f16` 约 0.0407 ms；
- 同数据类型下，向量化版本普遍更快，且**打包版本收益最大**：
  同一组里 `f16x8pack` 约 0.0147 ms，相对标量 `f16` 约 0.0407 ms 提升约 2.8 倍；
- FP32 的 `f32x4` 与大尺寸下的标量 `f32` 基本持平（0.1880 ms vs 0.1885 ms），
  说明该规模下 FP32 已经撞上带宽上限，减少访存指令条数不再带来额外收益；
- 小规模（如 `S=1024, K=1024`）时耗时差异变小甚至出现波动，
  因为 kernel 时间已经接近启动开销的量级；
- ReLU 每元素只有一次取最大值，计算量比 sigmoid（一次 `exp` + 一次除法）
  小得多，所以"访存优化 → 耗时下降"的因果关系在本模块里表现得最干净，
  可以把这里的耗时当作同类逐元素算子的**性能上界参照**。

## 后续可以尝试的优化方向

- 补上向量化版本的尾部（tail）分支，使任意 `N` 都安全
  （顺带消除 `f16x8_pack` 的漏算问题）；
- 去掉 `relu_f16x2_kernel` 里对 `y` 的冗余读取；
- 用 `__hmax2` 成对指令重写 `f16x2` / `f16x8`（unpack）版本，减少指令数；
- 引入 grid-stride loop，使固定大小的 grid 也能处理任意大 `N`；
- 用向量化 + `__ldg` / 只读缓存路径对比不同访存策略的差异；
- 对比不同 `block` 大小、不同向量宽度对带宽的实测影响；
- 与 `torch.relu` 对比时补充 `torch.nn.functional.relu`、`clamp_min(0)`
  等不同实现路径的性能差异。
