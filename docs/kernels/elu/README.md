# ELU（Exponential Linear Unit，指数线性单元）模块设计说明

## 模块目标

本模块用于演示 CUDA 中逐元素 **ELU（指数线性单元）** 的实现：

```text
elu(x) = { x,                    x > 0
         { alpha * (exp(x) - 1), x <= 0        （默认 alpha = 1.0）
```

同时展示同一个运算的不同优化写法：

1. 标量版本：一个线程处理一个元素；
2. 向量化版本（`float4` / `half2` / 128-bit 打包）：一个线程一次处理多个连续元素；
3. FP16 版本：用更小精度的数据类型，成倍降低访存字节数。

本模块与 elementwise、sigmoid、relu 三个模块是**姊妹示例**：6 个 kernel 版本
一一对应，区别只是把 `c[i] = a[i] + b[i]` / `y[i] = sigmoid(x[i])` /
`y[i] = max(0, x[i])` 换成 `y[i] = elu(x[i])`。前面模块讲过的线程编号、
向量化访存、FP16 打包等通用知识本模块不再重复。

ELU 在四个激活函数里位置很有意思：**正半轴和 ReLU 一样是恒等映射，负半轴
却像 sigmoid 一样饱和**——它把 ReLU 的"不饱和"和 sigmoid 的"平滑"各取一半，
用来缓解 ReLU 的"死亡神经元"问题（负半轴输出不是硬零而是缓降的负值，
均值更接近 0）。

最终目标不是追求极致性能，而是让学习者直观理解 CUDA 编程模型与访存优化思想。

## 本次改动范围

本模块参照 elementwise / sigmoid / relu 模块的做法，统一补充教学注释，
改动原则是**只加注释、不改 kernel 逻辑**，具体包含：

1. 先更新本文档（`docs/kernels/elu/README.md`）；
2. 给 `kernels/elu/elu.cu` 补充分节标题、调用链、grid/block 速查、
   ELU 数学性质说明，以及逐行行内注释；
3. 给 `kernels/elu/elu.py` 补充模块 docstring 与行内注释，并新增
   打印 PyTorch CUDA 扩展实际构建目录、以及打印当前 GPU 设备名的代码
   （对齐 `relu.py` / `sigmoid.py`）。

`elu.py` 的**运行行为保持不变**：官方对照仍是脚本里自己拼的
`torch_elu`（`torch.where` + `torch.exp`），结果打印仍保留 `f"{v:<12}"`
的左对齐补齐格式，因此 `kernels/elu/README.md` 中的测试输出样例无需改动。

`scripts/README.md` 中补充了 elu 模块的运行命令。

向量化版本（`f32x4` / `f16x2` / `f16x8` / `f16x8_pack`）缺少尾部（tail）分支的
问题，在本文档中已识别并说明，但**本次不修复**，留待后续单独处理。

## 涉及文件

| 文件 | 作用 |
| --- | --- |
| `kernels/elu/elu.cu` | CUDA kernel 定义 + PyTorch C++ 绑定（代码中已添加详细注释） |
| `kernels/elu/elu.py` | 用 `torch.utils.cpp_extension.load` 编译并执行基准测试 |
| `kernels/elu/README.md` | 模块使用说明与测试输出 |

## 注释导览

对照 elementwise / sigmoid / relu 模块，`elu.cu` 中的注释按下面的顺序组织：

| 位置 | 内容 |
| --- | --- |
| 文件头 | 模块公式、6 个版本概览、与其它激活函数模块的对应关系、学习重点 |
| 【ELU 曲线速览】 | ASCII 折线图、采样值、负半轴饱和 / 零均值 / 处处可导的讨论 |
| 宏定义段 | `reinterpret_cast` 向量访存宏的用法与对齐要求、`ALPHA` 的含义、为什么不需要 clamp |
| `elu` / `elu_half` | FP32 与 FP16 两套 `__device__ __forceinline__` 辅助函数的差异 |
| `elu_f32_kernel` | 调用链、grid/block 速查、三元选择与 NaN 语义、逐行注释 |
| `elu_f32x4_kernel` | `float4` 向量化访存、逐分量展开计算、缺少 tail 分支的风险 |
| `elu_f16_kernel` | FP16 常数转换开销、`__hgt` / `hexp` 选型 |
| `elu_f16x2_kernel` | `half2` 双分量计算 |
| `elu_f16x8_kernel` | unpack 写法：4 个 `half2` 完成 8 个元素 |
| `elu_f16x8_pack_kernel` | pack 写法：局部数组 + 128 位整体访存 |
| `TORCH_BINDING_ELU` | host 端宏展开细节与调用链 |
| `PYBIND11_MODULE` | 导出给 Python 的函数名 |

`TORCH_BINDING_ELU` 宏内部的注释使用 `/* ... */` 块注释而非 `//`：
宏定义每行以 `\` 续行，而 `//` 会把该行末尾的续行符一起注释掉，宏定义会当场断开。
`/* ... */` 块注释在预处理阶段被替换为空格，不会影响续行。

## CUDA 核心概念回顾

### 1. 线程组织

和 elementwise / sigmoid / relu 一样，kernel 启动后每个线程通过以下内置变量确定自己的位置：

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

### 2. ELU 的数学性质

```text
elu(x) = { x,                    x > 0
         { alpha * (exp(x) - 1), x <= 0        （本模块 alpha = 1.0）
```

函数曲线（ASCII 示意图，`x ∈ [-3, 3]`，`alpha = 1`，虚线为渐近线 `y = -1`）：

```text
 3.0 ┤                          ***
     ┤                       ***
 2.0 ┤                    ***
     ┤                 ***
 1.0 ┤              ***
     ┤           ***
 0.0 ┤**********+
     ┤      ****
-0.5 ┤  ****
-1.0 ┤*····························  ← 渐近线 y = -alpha
     └──────────────────────────────→ x
     -3   -2   -1    0    1    2    3
```

采样值：`-3 → -0.9502`、`-2 → -0.8647`、`-1 → -0.6321`、`0 → 0`、
`1 → 1`、`2 → 2`、`3 → 3`。

读图要点：

- **正半轴是恒等映射**：`x > 0` 时 `elu(x) = x`，这一半和 ReLU 完全一致，
  同样不饱和、梯度恒为 1；
- **负半轴是指数饱和**：`x <= 0` 时 `elu(x) = alpha * (exp(x) - 1)`，
  单调递增、有界，`x → -∞` 时极限是 `-alpha`（本模块即 -1），
  但**永远取不到** `-alpha`，所以值域是开区间 `(-alpha, +∞)`；
- **在 `x = 0` 处连续且可导**：左导数 `exp(0) = 1`、右导数 `1`，
  两侧相等，所以 `alpha = 1` 时 ELU 在原点**一阶可导**（这正好补上了
  ReLU 在 0 点不可导的短板）；`elu(0)` 走的是 `else` 分支，
  算出来是 `exp(0) - 1 = 0`，与正半轴上界严格衔接；
- **负半轴导数不为 0**：`elu'(x) = exp(x) ∈ (0, 1]`，
  这与 ReLU 的"负半轴梯度恒为 0"形成关键对比——ELU 不会出现
  **死亡 ReLU（dead ReLU）**，负半轴仍然能把梯度回传；
- **输出均值更接近 0**：负半轴输出的是负值而不是硬零，
  缓解了 ReLU 输出恒非负、把下一层偏置整体推偏的问题；
- **代价**：负半轴要算一次指数，比 ReLU 贵；因此本模块的性能特征
  介于 ReLU（纯访存瓶颈）和 sigmoid（计算偏重）之间。

与同族算子的对照（下文会反复引用这张表）：

| 对比项 | sigmoid | relu | elu（alpha = 1） |
| --- | --- | --- | --- |
| 公式 | `1/(1+exp(-x))` | `max(0, x)` | `x > 0 ? x : exp(x) - 1` |
| 值域 | `(0, 1)`，两端饱和 | `[0, +∞)`，仅负端截断 | `(-1, +∞)`，仅负端饱和 |
| `x = 0` 处 | 可导，导数 0.25 | 不可导（左 0 右 1） | 可导，导数 1 |
| 负半轴梯度 | 趋近 0（饱和） | 恒为 0（死亡 ReLU） | `exp(x) > 0`（衰减但不为 0） |
| 每元素计算 | 一次 `exp` + 一次加法 + 一次除法 | 一次取最大值 | 一次比较 + 一次 `exp`（仅负半轴） |
| 是否需要 clamp | 需要（见 `docs/kernels/sigmoid/README.md`） | 不需要 | 不需要（见下节） |

### 3. 为什么 ELU 不需要溢出保护（与 sigmoid 的关键差异）

sigmoid 必须先 `clamp` 再 `exp`，否则 `exp(-x)` 会溢出（见
`docs/kernels/sigmoid/README.md` 中的 `MAX_EXP_F32` / `MAX_EXP_F16` 说明）。
ELU 完全没有这个问题，原因在于**指数只出现在负半轴**：

| 输入情况 | `elu` 的结果 |
| --- | --- |
| `x` 是很大的正数（如 `3.4e38`） | 走 `x > 0` 分支，原样返回，不产生新数值 |
| `x` 是很大的负数（如 `-3.4e38`） | `exp(x)` 下溢成 0，结果是 `alpha * (0 - 1) = -alpha`，正好是极限值 |
| `x` 是 `0` | `exp(0) - 1 = 0`，与正半轴严格连续 |
| `x` 是 `+inf` / `-inf` | `+inf` / `-alpha` |

即使编译器把三元表达式编译成"两条分支都算、再按谓词选择"（常见的谓词化写法），
正半轴上算出的 `expf(大正数) = inf` 也只会被丢掉，不会污染结果——
因为此时选择的是 `x` 本身。

与 ReLU 的 NaN 语义差异也值得对照：ReLU 用的 `fmaxf(NaN, 0) = 0` 会**把 NaN 吞掉**，
而 ELU 的分支条件是 `x > 0`，NaN 比较结果为假，于是走进 `exp(NaN) - 1`，
自然把 NaN 传播到输出，**与 PyTorch 的 `torch.nn.functional.elu` 行为一致**。

### 4. FP16 精度提醒

FP32 尾数 23 位，最小刻度约 `2^-23 ≈ 0.00000012`（约 7 位有效数字）；
FP16 尾数只有 10 位，最小刻度为 `2^-10 = 1/1024` ≈ `0.00098`
（约 3~4 位有效数字，保守按约 3 位）。

ELU 在 FP16 下有**两处**需要额外留意的地方：

1. **`exp(x) - 1` 的相减抵消**：当 `x` 是接近 0 的负数时，
   `exp(x) ≈ 1`，两者相减后有效位数大量丢失。例如 `x = -0.001` 时
   `exp(x) ≈ 0.999`，`0.999 - 1 = -0.001`——在 10 位尾数下这个差值的
   相对误差会被放大。这是公式本身的性质，不是 kernel 的 bug；
2. **负半轴的饱和**：`x` 小到大约 -11 时 `hexp(x)` 已经下溢成 0，
   结果直接取到 `-alpha`（FP32 则要到 -88 左右才下溢）。
   本模块基准测试用 `torch.randn` 生成输入（几乎都落在 ±5 内），
   不会触发这一饱和。

另外，FP16 的最大规格数只有 65504，`torch.randn` 生成的输入远小于它，
不会触发上溢。

## Kernel 设计

所有 kernel 都遵守一个约定：**把输入输出都当作长度为 `N` 的一维数组处理**。
二维矩阵 `(S, K)` 在启动端被展平为 `N = S * K`。

### 0. 两个辅助函数：`elu` 与 `elu_half`

ELU 的公式在 FP32 与 FP16 下写法完全不同，因此本模块抽出了两个
`__device__ __forceinline__` 辅助函数：

```c
// FP32：三元选择 + 标量 expf
__device__ __forceinline__ float elu(float x) {
  return x > 0.f ? x : ALPHA * (expf(x) - 1.f);
}

// FP16：half 没有隐式类型提升，比较 / 乘法 / 减法 / 指数都要用 half 内建函数
__device__ __forceinline__ half elu_half(half x) {
  return __hgt(x, __float2half(0.f))
             ? x
             : __hmul(__float2half(ALPHA), __hsub(hexp(x), __float2half(1.f)));
}
```

- `__device__` 表示"只能在设备（GPU）代码里调用"；`__forceinline__` 要求
  编译器强制内联，避免函数调用开销（逐元素算子每个线程都要调用一次）；
- `ALPHA` 是本文件顶部的宏（`#define ALPHA 1.0f`），PyTorch 的
  `torch.nn.functional.elu` 默认也是 `alpha = 1.0`，两者默认值一致；
- FP16 版本对应关系：`>` → `__hgt`、`*` → `__hmul`、`-` → `__hsub`、
  `expf` → `hexp`。如果直接写普通运算符，在 "
  `-U__CUDA_NO_HALF_OPERATORS__` 等宏被恢复（见 `elu.py` 的编译选项）的
  情况下虽然也能编译，但语义上仍以显式内建函数最清晰；
- 命名上 FP32 版直接叫 `elu`、FP16 版叫 `elu_half`，与 swish 模块的
  `swish` / `swish_half` 命名习惯一致（本次不重命名）。

### 1. `elu_f32_kernel`（FP32 标量版）

```c
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N)
  y[idx] = elu(x[idx]);
```

- 一个线程只算一个元素；
- `if (idx < N)` 处理 `N` 不能被 block 大小整除时"多启动"的线程；
- 正确性最直观，是后面所有版本的基准。

### 2. `elu_f32x4_kernel`（FP32 向量化版）

每个线程连续处理 **4 个 float（16 字节）**：

```c
int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
if (idx < N) {
  float4 reg_x = FLOAT4(x[idx]);
  float4 reg_y;
  reg_y.x = elu(reg_x.x);
  ...
  FLOAT4(y[idx]) = reg_y;
}
```

- 将一段连续内存 reinterpret 成 `float4`，用一次加载/存储完成 16 字节搬运，
  4 次 4 字节访存被压缩成 1 次 16 字节访存；
- 与 elementwise 不同，ELU 和 sigmoid / relu 一样是**单输入**算子：
  只需读 `x` 一份数据，没有 `b` 侧的第二路 load；
- "向量化"省的只是访存指令：ELU 的计算仍要按 `x/y/z/w` 四个分量逐条展开，
  每个分量都可能触发一次 `expf`，因此向量化的收益不如纯访存型的 ReLU 明显
  （实测中 `f32x4` 与 `f32` 基本持平，见文末"性能观察"）。

### 3. `elu_f16_kernel`（FP16 标量版）

```c
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N)
  y[idx] = elu_half(x[idx]);
```

- 数据类型为 `half`（2 字节），字节数减半，是后续 FP16 向量化的基础；
- 计算全部在 `elu_half` 里完成，kernel 体只剩线程编号和访存，
  与其它模块的标量版保持同一种排版；
- 常数 0 与 1、`ALPHA` 都用 `__float2half` 就地转换：参数是字面量时
  编译器会直接折叠成 half 常量，不产生运行时开销。

### 4. `elu_f16x2_kernel`（FP16 每线程 2 元素）

- 一次用 `half2`（2 个 half = 4 字节）完成两个元素的搬运；
- `half2` 只有 `x`、`y` 两个分量，两个分量各自独立调用 `elu_half`；
- 与 relu 的 `f16x2` 不同，ELU 没有 `__hmax2` 那样的成对指令可用
  （`hexp` 没有 `hexp2` 版本），所以这里只能逐分量计算，
  不能像 ReLU 的 pack 版本那样把循环步长写成 2。

### 5. `elu_f16x8_kernel`（FP16 每线程 8 元素，unpack 写法）

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
- 结果是 4 个 `half2`，分别写回 `y[idx+0/2/4/6]`；
- 注意 4 次 `HALF2(x[...])` 读取是**无条件执行**的，越界判断只出现在写回处。

### 6. `elu_f16x8_pack_kernel`（128-bit 打包版）

```c
half pack_x[8], pack_y[8]; // 8x16 bits=128 bits.
LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); // load 128 bits
#pragma unroll
for (int i = 0; i < 8; i++) {
  pack_y[i] = elu_half(pack_x[i]);
}
LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
```

- 把 8 个 half（正好 16 字节 = 128 位）装入局部数组；
- 借助 reinterpret 成 `float4` 的 `LDST128BITS` 宏，让一次访存指令读写完整的
  128 位，比 unpack 版本的多次 `half2` 访存更省指令；
- 局部数组在 PTX 中通常对应 `.local` 空间（可寻址），
  但配合 `#pragma unroll` 与编译期常数下标，编译器有机会把它优化到寄存器里，
  从而避免真正访问 local memory；
- 循环步长是 `i++` 而不是 ReLU pack 版的 `i += 2`：**ELU 没有成对指令**，
  8 个元素只能逐个 half 计算。

## 边界约束（已识别，本次不修复）

elementwise 的向量化版本用 `(idx + 3) < N` 判断整段元素是否齐全，
并配了尾部标量回退分支；本模块的向量化版本**两个都没有**，
越界判断只检查了段的起始下标：

| kernel | 现有判断 | `N` 不是向量宽度的整数倍时会发生什么 |
| --- | --- | --- |
| `elu_f32x4_kernel` | `idx < N`（在 load 之前） | 最后一个 block 的部分线程会把 `idx+1..idx+3` 写到 `y` 合法范围之外（越界写 1~3 个 float） |
| `elu_f16x2_kernel` | `idx < N`（在 load 之前） | 同上，越界写 1 个 half |
| `elu_f16x8_kernel` | `(idx + 0/2/4/6) < N`（在 store 处） | 同上，越界写 1 个 half |
| `elu_f16x8_pack_kernel` | `(idx + 7) < N`（在 store 处） | 方向相反：末尾不足 8 个的元素被整块**丢弃**（漏算），但不会越界 |

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

### 与 PyTorch 的一致性

- **正半轴**：kernel 直接返回 `x`，与 PyTorch 完全一致；
- **负半轴**：kernel 用 `expf(x) - 1`，PyTorch 的 CUDA 实现同样走 `expm1`
  风格的指数计算，两者在 FP32 下按位级接近（`kernels/elu/README.md`
  的测试输出里 `out_f32` 与 `out_f32_th` 数值一致）；
- **NaN**：kernel 会因为 `x > 0` 为假而走进指数分支，NaN 被传播，
  与 PyTorch 一致（这一点和 relu 模块的 `fmaxf` 吞 NaN 不同）；
- **FP16**：误差来自"输入由 FP32 就近取整成 FP16"以及"`exp(x) - 1`
  的相减抵消"，属于预期精度损失，测试输出里可以看到
  `-0.18413025`（FP32）对 `-0.18408203`（FP16）这类差异。

## 启动配置（host 端）

启动端把每个 block 负责的元素数固定为 **256**：

```c
dim3 block(256 / (n_elements));   // 每个线程负责 n_elements 个元素
dim3 grid((N + 256 - 1) / 256);   // 向上取整
```

各版本的参数对应关系：

| 版本 | `n_elements` | block 线程数 | 每 block 处理元素数 |
| --- | ---: | ---: | ---: |
| `elu_f32` | 1 | 256 | 256 |
| `elu_f32x4` | 4 | 64 | 256 |
| `elu_f16` | 1 | 256 | 256 |
| `elu_f16x2` | 2 | 128 | 256 |
| `elu_f16x8` | 8 | 32 | 256 |
| `elu_f16x8_pack` | 8 | 32 | 256 |

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

`elu.cu` 通过 `TORCH_BINDING_ELU` 宏批量生成 6 个 host 函数，
再经 `PYBIND11_MODULE` 暴露为 Python 可调用模块，主要工作：

1. 用 `CHECK_TORCH_TENSOR_DTYPE` 检查输入输出张量数据类型；
2. 根据维度/形状计算 `grid` 与 `block`；
3. 用 `reinterpret_cast<element_type *>(x.data_ptr())` 拿到裸指针；
4. 以 `kernel<<<grid, block>>>` 启动内核。

### 调用链（以 `elu_f32` 为例）

```text
elu.py
  └─ lib.elu_f32(x, y)                       # Python 调用（pybind11）
      └─ elu_f32(torch::Tensor x, y)         # C++ host 包装函数（宏生成）
          ├─ CHECK_TORCH_TENSOR_DTYPE(x/y)   # 校验 dtype 必须是 float32
          ├─ 计算 dim3 block / dim3 grid
          └─ elu_f32_kernel<<<grid, block>>>(...)   # CUDA 启动语法
              └─ cudaLaunchKernel(...)       # nvcc 生成，交给 CUDA runtime
                  └─ GPU 各线程并行执行 kernel 函数体
```

要点：

- Python 调用的 `elu_f32` 运行在 CPU 上，负责类型检查、计算 `grid` / `block`；
- `elu_f32_kernel` 运行在 GPU 上，必须用 `<<<>>>` 启动，不能普通函数调用；
- 启动后 GPU 的每个线程都会执行一遍 kernel 函数体，靠 `threadIdx` / `blockIdx`
  区分自己负责的下标 `idx`；
- `CHECK_TORCH_TENSOR_DTYPE` 的报错文案是 `"Tensor dtype must be ..."`，
  与 relu 模块的 `"values must be ..."` 略有不同，本模块保留原样。

## 测试脚本

`elu.py` 流程：

1. 用 `torch.utils.cpp_extension.load` 现场编译 `elu.cu`；
2. 对 `S ∈ {1024, 2048, 4096}`、`K ∈ {1024, 2048, 4096}` 组合生成随机张量；
3. `run_benchmark` 先做 warmup，再运行 1000 次取平均；
4. 依次对比 6 个自定义 kernel 与脚本内自定义的 `torch_elu` 的正确性和耗时。

脚本在 `load(...)` 完成后会打印 PyTorch CUDA 扩展的实际构建目录，
便于确认 `elu_lib.so` 的存放位置；随后用
`print(torch.cuda.get_device_name())` 打印当前 GPU 设备名，
便于对照不同显卡上的耗时数据。

`torch_elu` 是本脚本自己拼出来的对照实现（**不是** PyTorch 的融合算子）：

```python
def torch_elu(x, out=None):
    if out is None:
        return torch.where(x > 0, x, 1.0 * (torch.exp(x) - 1))
    else:
        torch.where(x > 0, x, 1.0 * (torch.exp(x) - 1), out=out)
        return out
```

- 数学上等价于 `alpha = 1` 的 `torch.nn.functional.elu`；
- 但它由 `gt`、`exp`、`sub`、`mul`、`where` 等多个 elementwise 算子组成，
  读写显存多次，所以耗时明显高于单个融合 kernel——
  这也是 `kernels/elu/README.md` 里 `out_f32_th` 一栏偏慢的原因；
- **公平的比较对象应该是 `torch.nn.functional.elu`**（单个融合 kernel），
  本次保持脚本原样不替换，仅在文档中说明这个差异。

打印对照结果前，`run_benchmark` 会通过
`out.flatten().detach().cpu().numpy().tolist()[:2]` 把 GPU 上的结果张量
拉平、断开可能的梯度追踪（本脚本已关闭 autograd，属于保险写法）、拷回 CPU、
转成 Python 列表，并只取前 2 个元素用于人工核对；本模块还额外用
`f"{v:<12}"` 把每个数值左对齐补齐到 12 个字符，所以
`kernels/elu/README.md` 里的输出带引号和多余空格，属正常格式。

判断正确性要看同组里 6 个版本与 `out_f32_th` 是否**数值一致**：
输入随机、正负混合，所以 `out_f32` 一栏既可能出现正数（正半轴恒等映射），
也可能出现 `-0.xxxx`（负半轴的指数段），都是正常的。

## 性能观察与结论

从 `kernels/elu/README.md` 的测试输出可总结：

- **FP16 明显快于 FP32**：访存字节数减半，例如 `S=4096, K=4096` 时
  `f32` 约 0.146 ms、`f16` 约 0.033 ms，接近 4.4 倍；
- **FP16 内部越宽越快**：同一组里 `f16x8pack` 约 0.0169 ms、
  `f16x8` 约 0.0181 ms、`f16x2` 约 0.0317 ms、`f16` 约 0.0329 ms，
  反映"访存指令条数 → 耗时"的直接关系；
- **FP32 向量化收益有限**：`f32x4`（0.1461 ms）与 `f32`（0.1459 ms）几乎持平。
  因为 FP32 下每个元素都要读写 4 字节、且负半轴要算 `expf`，
  规模足够大时已经接近带宽上限，减少访存指令条数不再带来额外收益；
- **对照实现偏慢是预期结果**：`out_f32_th` 在 `S=4096, K=4096` 时约 0.762 ms，
  约是自定义 kernel 的 5 倍，原因是它由多个算子拼成、多次读写显存
  （不是与 `torch.nn.functional.elu` 的公平对比，见上文说明）；
- 小规模（如 `S=1024, K=1024`）时各版本耗时差异变小甚至出现波动，
  因为 kernel 时间已经接近启动开销的量级；
- ELU 的计算量介于 ReLU 与 sigmoid 之间：正半轴几乎免费，
  负半轴要算一次指数，所以它的加速曲线既有 ReLU 那种"带宽受限"的特征，
  也有 sigmoid 那种"计算占一部分"的特征。

## 后续可以尝试的优化方向

- 补上向量化版本的尾部（tail）分支，使任意 `N` 都安全
  （顺带消除 `f16x8_pack` 的漏算问题）；
- 把脚本里的对照实现从 `torch.where` 拼装换成
  `torch.nn.functional.elu`，得到真正公平的性能对比；
- 用 `__expf`（快速指数）替换 `expf` 观察精度与速度的取舍
  （本模块已开 `--use_fast_math`，实际已经走到快速指令路径）；
- 引入 `expm1f(x)` 替代 `expf(x) - 1.f`，改善 `x → 0` 附近
  相减抵消造成的相对误差（对应 FP16 的 `hexp` 版本则要考虑成本）；
- 把 `alpha` 从编译期宏改成运行期参数（kernel 增加一个 `float alpha` 入参），
  对齐 `torch.nn.functional.elu` 的完整签名；
- 引入 grid-stride loop，使固定大小的 grid 也能处理任意大 `N`；
- 对比不同 `block` 大小、不同向量宽度对带宽的实测影响。
