# Swish（Swish / SiLU，自门控激活函数）模块设计说明

## 模块目标

本模块用于演示 CUDA 中逐元素 **Swish**（在 PyTorch 里叫 **SiLU**，
Sigmoid Linear Unit）的实现：

```text
swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
```

> 说明：swish 与 silu 是**同一个函数**的两个名字。
> Google 在 2017 年的论文里把它叫 Swish（`x * sigmoid(beta * x)`，
> `beta = 1` 时即本模块的形式），PyTorch 沿用了 SiLU 这个叫法
> （`torch.nn.SiLU` / `torch.nn.functional.silu`）。本模块文件沿用仓库里
> 已有的 `swish` 命名，文档中两者不加区分。

同时展示同一个运算的不同优化写法：

1. 标量版本：一个线程处理一个元素；
2. 向量化版本（`float4` / `half2` / 128-bit 打包）：一个线程一次处理多个连续元素；
3. FP16 版本：用更小精度的数据类型，成倍降低访存字节数。

本模块与 elementwise、sigmoid、relu、elu、gelu 模块是**姊妹示例**：6 个 kernel
版本一一对应，区别只是把 `c[i] = a[i] + b[i]` / `y[i] = sigmoid(x[i])` /
`y[i] = max(0, x[i])` 换成 `y[i] = swish(x[i])`。前面模块讲过的线程编号、
向量化访存、FP16 打包等通用知识本模块不再重复，下面重点说明 Swish 特有的三点：

- **它把 sigmoid 用在了自己身上**：公式里既出现 `x`，又出现 `sigmoid(x)`，
  所以本模块可以看作是 sigmoid 模块的"进阶版"——同样的 `exp`，
  多一步加法、除法和乘法；
- **写成了一个除法**：kernel 里没有真的先算 sigmoid 再乘，
  而是用等价的 `x / (1 + exp(-x))` 一次算完，省掉一次中间变量；
- **不需要 clamp**：`exp(-x)` 溢出成 `inf` 时结果恰好是数学极限 0，
  不会出现 NaN（这一点与 gelu 的 `hexp` 拼接式 tanh 形成鲜明对比）。

最终目标不是追求极致性能，而是让学习者直观理解 CUDA 编程模型与访存优化思想。

## 本次改动范围

本模块参照 elementwise / sigmoid / relu 模块的做法，统一补充教学注释，
改动原则是**只加注释、不改 kernel 逻辑**，具体包含：

1. 先更新本文档（`docs/kernels/swish/README.md`）；
2. 给 `kernels/swish/swish.cu` 补充分节标题、调用链、grid/block 速查、
   Swish 数学性质说明，以及逐行行内注释；
3. 给 `kernels/swish/swish.py` 补充模块 docstring 与行内注释，并新增
   打印 PyTorch CUDA 扩展实际构建目录、以及打印当前 GPU 设备名的代码
   （对齐 `relu.py` / `sigmoid.py`）。

`swish.py` 的**运行行为保持不变**：官方对照仍是脚本里自己拼的 `torch_swish`
（`torch.sigmoid` + `mul_`），结果打印仍保留 `f"{v:<12}"` 的左对齐补齐格式，
因此 `kernels/swish/README.md` 中的测试输出样例无需改动。

`scripts/README.md` 中补充了 swish 模块的运行命令。

## 涉及文件

| 文件 | 作用 |
| --- | --- |
| `kernels/swish/swish.cu` | CUDA kernel 定义 + PyTorch C++ 绑定（代码中已添加详细注释） |
| `kernels/swish/swish.py` | 用 `torch.utils.cpp_extension.load` 编译并执行基准测试 |
| `kernels/swish/README.md` | 模块使用说明与测试输出 |

## 注释导览

对照 elementwise / sigmoid / relu 模块，`swish.cu` 中的注释按下面的顺序组织：

| 位置 | 内容 |
| --- | --- |
| 文件头 | 模块公式、swish 与 silu 的命名关系、6 个版本概览、学习重点 |
| 【Swish 曲线速览】 | ASCII 折线图、采样值、非单调 / 负向凹陷 / 与 ReLU、GELU 的对照 |
| 宏定义段 | `reinterpret_cast` 向量访存宏的用法与对齐要求、为什么不需要 clamp |
| `swish` / `swish_half` | FP32 与 FP16 两套 `__device__ __forceinline__` 辅助函数的差异 |
| `swish_f32_kernel` | 调用链、grid/block 速查、为什么写成除法、逐行注释 |
| `swish_f32x4_kernel` | `float4` 向量化访存、逐分量展开计算、缺少 tail 分支的风险 |
| `swish_f16_kernel` | FP16 常数转换开销、`hexp` / `__hdiv` 选型 |
| `swish_f16x2_kernel` | `half2` 双分量计算 |
| `swish_f16x8_kernel` | unpack 写法：4 个 `half2` 完成 8 个元素 |
| `swish_f16x8_pack_kernel` | pack 写法：局部数组 + 128 位整体访存 |
| `TORCH_BINDING_SWISH` | host 端宏展开细节与调用链 |
| `PYBIND11_MODULE` | 导出给 Python 的函数名 |

`TORCH_BINDING_SWISH` 宏内部的注释使用 `/* ... */` 块注释而非 `//`：
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

### 2. Swish 的数学性质

```text
swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
```

函数曲线（ASCII 示意图，`x ∈ [-3, 3]`）：

```text
 3.0 ┤                          ***
     ┤                       ***
 2.0 ┤                    ***
     ┤                 ***
 1.0 ┤              ***
     ┤           ***
 0.0 ┤*******+
     ┤      ***
-0.28┤   *                       ← 负向小凹陷，min ≈ -0.2785（x ≈ -1.279）
     └──────────────────────────────→ x
     -3   -2   -1   0   1   2   3
```

采样值：`-3 → -0.1423`、`-2 → -0.2384`、`-1.279 → -0.2785`、`-1 → -0.2689`、
`0 → 0`、`1 → 0.7311`、`2 → 1.7616`、`3 → 2.8577`、`4 → 3.9281`。

读图要点：

- **正半轴近似恒等映射**：`x` 稍大（约 3 以上）后 `sigmoid(x) ≈ 1`，
  `swish(x) ≈ x`，与 ReLU / ELU / GELU 的正半轴一致；
- **负半轴不是硬零**：`x < 0` 时输出是小负数，`x → -∞` 时趋近 0
  （因为 `x * 0` 里 `sigmoid(x)` 衰减得比 `x` 增长更快），
  最小值约 `-0.2785`（出现在 `x ≈ -1.279`），所以 Swish **不是单调函数**；
- **它是"自门控"的**：`sigmoid(x)` 扮演门控——`x` 为正时门开（信号通过），
  `x` 为负时门关（信号被压制）。这就是 Swish 与 GELU 被称为
  "self-gated / 平滑 ReLU" 家族的原因；
- **在 `x = 0` 处光滑可导**，导数 `swish'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))`，
  在 `x = 0` 处等于 0.5。注意：**在 `x = 0` 处导数不为 0**，
  这与 ReLU 的次梯度取 0 不同；
- **与 GELU 的关系**：两者形状非常接近（都是"先下凹再单调上升"），
  GELU 可以看成"用正态 CDF 当门控"，Swish 用的是 sigmoid 门控；
  Swish 的凹陷更深（-0.2785 对 GELU 的 -0.1700），
  而计算上 Swish 更便宜（一次 `exp` + 一次加法 + 一次除法 + 一次乘法，
  没有 `tanh` 里那套三次多项式）。

与同族算子的对照（下文会反复引用这张表）：

| 对比项 | relu | elu（alpha = 1） | gelu（tanh 近似） | swish / silu |
| --- | --- | --- | --- | --- |
| 公式 | `max(0, x)` | `x > 0 ? x : exp(x) - 1` | `0.5x(1 + tanh(k(x + 0.044715x^3)))` | `x / (1 + exp(-x))` |
| 值域 | `[0, +∞)` | `(-1, +∞)` | `(-0.17, +∞)` | `(-0.2785, +∞)` |
| 最小值位置 | `x <= 0` 全部取 0 | 极限 -1（取不到） | `x ≈ -0.752` | `x ≈ -1.279` |
| `x = 0` 处 | 不可导 | 可导，导数 1 | 光滑可导 | 光滑可导，导数 0.5 |
| 每元素计算 | 一次取最大值 | 一次比较 + 一次 `exp` | 多次乘加 + 一次 tanh | 一次 `exp` + 加法 + 除法 + 乘法 |
| 是否需要 clamp | 不需要 | 不需要 | **需要**（见 gelu 文档） | 不需要（见下节） |

### 3. 为什么 Swish 不需要溢出保护

本模块的公式里有个除法 `x / (1 + exp(-x))`，但**不需要**像 sigmoid / gelu
那样先 clamp 输入。原因是 `exp` 的两个极端都对应正确的数学极限：

| 输入情况 | `exp(-x)` 的值 | `swish(x)` 的结果 | 说明 |
| --- | --- | --- | --- |
| `x` 是很大的正数（如 `88`） | `exp(-88) ≈ 6e-39`，接近 0 | `x / 1 = x` | 上界正确：`sigmoid(88) ≈ 1` |
| `x` 是很大的负数（如 `-88`） | `exp(88)` 溢出成 `inf` | `x / inf = -0` | 下界正确：数学极限就是 `x * 0` |
| `x = 0` | `exp(0) = 1` | `0 / 2 = 0` | 恰好过原点 |
| `x` 是 `+inf` / `-inf` | `0` / `inf` | `+inf` / `NaN` | 与 PyTorch 行为一致（`-inf * 0` 是 NaN） |
| `x` 是 `NaN` | `NaN` | `NaN` | NaN 正常传播，与 PyTorch 一致 |

关键在第二行：`exp(-x)` 溢出得到的 `inf` 恰好对应 `sigmoid(-∞) = 0`，
而 `x / inf = 0` 正是极限值。**溢出在这里是"无害"的**，
不像 gelu 那样会变成 `inf / inf = NaN`。

这也是一个很好的对照实验：同样的 `exp`，放在分母上就安全，
放在分子上做减法（`exp(2t) - 1`）再自己除自己就不安全。
物理意义是"数值溢出发生在极限附近时，结果依然正确"。

### 4. FP16 精度提醒

FP32 尾数 23 位，最小刻度约 `2^-23 ≈ 0.00000012`（约 7 位有效数字）；
FP16 尾数只有 10 位，最小刻度为 `2^-10 = 1/1024` ≈ `0.00098`
（约 3~4 位有效数字，保守按约 3 位）。

Swish 在 FP16 下有两处需要注意：

1. **分母的"吞掉小量"**：`1 + hexp(-x)` 里，当 `hexp(-x) < 2^-11`
   （约 4.9e-4）时，一半的加法会把这一项直接舍掉，`1 + ε` 变成 `1.0`，
   于是 `sigmoid(x)` 被四舍五入成 1，`swish(x)` 退化成 `x`。
   实测 `x = 11` 时 `swish_f16` 返回 `11.0`，与 PyTorch 一致；
   这是 half 精度的固有限制，不是 kernel 的 bug；
2. **负方向的饱和**：`x <= -11.09` 时 `hexp(-x)` 溢出成 `inf`，
   `__hdiv(1, inf) = 0`，于是 `swish_f16(-12) = -0.0`；
   而 PyTorch（内部用 float 计算 sigmoid）给出的是 `-7.37e-05`。
   两者的**绝对误差只有 7e-5**，通常无感，但确实是一个可测量的差异，
   详见"边界约束"节。

另外，FP16 的最大规格数只有 65504，`torch.randn` 生成的输入远小于它，
不会触发输入本身的上溢。

## Kernel 设计

所有 kernel 都遵守一个约定：**把输入输出都当作长度为 `N` 的一维数组处理**。
二维矩阵 `(S, K)` 在启动端被展平为 `N = S * K`。

### 0. 两个辅助函数：`swish` 与 `swish_half`

```c
// FP32：一个除法搞定，省掉"先算 sigmoid 再乘"的中间变量
__device__ __forceinline__ float swish(float x) {
  return x / (1.0f + expf(-x));
}

// FP16：half 没有隐式类型提升，加法 / 除法 / 乘法 / 指数都要用 half 内建函数
__device__ __forceinline__ half swish_half(half x) {
  return __hmul(x, __hdiv(__float2half(1.0f),
                          __hadd(__float2half(1.0f), hexp(__hneg(x)))));
}
```

- `__device__` 表示"只能在设备（GPU）代码里调用"；`__forceinline__` 要求
  编译器强制内联，避免函数调用开销（逐元素算子每个线程都要调用一次）；
- FP32 版**直接写成除法**：`x * (1 / (1 + exp(-x)))` 或
  `x * sigmoid(x)` 都等价，但一次除法再加一次乘法，不如
  `x / (1 + exp(-x))` 只做一次除法省事，寄存器压力也更小；
- FP16 版对应关系：`/` → `__hdiv`、`+` → `__hadd`、`*` → `__hmul`、
  `-x` → `__hneg`、`expf` → `hexp`。这里刻意写成
  `__hmul(x, __hdiv(1, 1 + hexp(-x)))`——与 FP32 版的除序相反，
  但数学上等价（`x / d == x * (1 / d)`）；
- 命名上 FP32 版直接叫 `swish`、FP16 版叫 `swish_half`，
  与 elu 模块的 `elu` / `elu_half` 命名习惯一致。

### 1. `swish_f32_kernel`（FP32 标量版）

```c
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N)
  y[idx] = swish(x[idx]);
```

- 一个线程只算一个元素；
- `if (idx < N)` 处理 `N` 不能被 block 大小整除时"多启动"的线程；
- 计算全部落在 `swish` 里，kernel 体只剩线程编号和访存，
  与其它模块的标量版保持同一种排版；
- 正确性最直观，是后面所有版本的基准。

### 2. `swish_f32x4_kernel`（FP32 向量化版）

每个线程连续处理 **4 个 float（16 字节）**：

```c
int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
if (idx < N) {
  float4 reg_x = FLOAT4(x[idx]);
  float4 reg_y;
  reg_y.x = swish(reg_x.x);
  ...
  FLOAT4(y[idx]) = reg_y;
}
```

- 一次 16 字节的向量 load/store 代替 4 次 4 字节访存，减少访存指令条数；
- 与 elementwise 不同，Swish 和 sigmoid / relu / elu / gelu 一样是**单输入**
  算子：只需读 `x` 一份数据，没有 `b` 侧的第二路 load；
- "向量化"省的只是访存指令：计算仍要按 `x/y/z/w` 四个分量逐条展开，
  每个分量都要算一次 `expf` 和一次除法，因此向量化的收益有限
  （实测 `f32x4` 比 `f32` 快一点，但远达不到 4 倍，见文末"性能观察"）；
- 与 sigmoid 模块的 `f32x4` 相比，这里少了"先 clamp"的四行，
  多了一次乘法（`x * sigmoid`），指令数量相当。

### 3. `swish_f16_kernel`（FP16 标量版）

```c
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N)
  y[idx] = swish_half(x[idx]);
```

- 数据类型为 `half`（2 字节），字节数减半，是后续 FP16 向量化的基础；
- 计算全部在 `swish_half` 里完成，kernel 体只剩线程编号和访存；
- half 的除法要比乘法贵（`__hdiv` 内部是一段较长的指令序列），
  写 `x * (1 / d)` 而不是 `x / d` 也是同样的考虑——
  反正都要算一次 `1 / d`，不如用乘法把结果乘回去。

### 4. `swish_f16x2_kernel`（FP16 每线程 2 元素）

- 一次用 `half2`（2 个 half = 4 字节）完成两个元素的搬运；
- `half2` 只有 `x`、`y` 两个分量，两个分量各自独立调用 `swish_half`；
- 与 relu 的 `f16x2` 不同，Swish 没有成对指令可用
  （既没有 `hexp2` 也没有成对的除法），所以这里只能逐分量计算，
  不能像 ReLU 的 pack 版本那样把循环步长写成 2。

### 5. `swish_f16x8_kernel`（FP16 每线程 8 元素，unpack 写法）

- 每个线程处理 8 个 half，拆成 4 个 `half2`（`idx+0/2/4/6`）分别处理；
- 这里的 "unpack" 指：数据在内存里本就是连续的 8 个 half，
  本 kernel 不把它整体打包搬运，而是按 `half2` 逐段读、逐段写；
- 相比 f16x2，每个线程做更多工作，能摊薄索引计算等固定开销；
- 结果是 4 个 `half2`，分别写回 `y[idx+0/2/4/6]`；
- 注意 4 次 `HALF2(x[...])` 读取是**无条件执行**的，越界判断只出现在写回处。

### 6. `swish_f16x8_pack_kernel`（128-bit 打包版）

```c
half pack_x[8], pack_y[8]; // 8x16 bits=128 bits.
LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); // load 128 bits
#pragma unroll
for (int i = 0; i < 8; i++) {
  pack_y[i] = swish_half(pack_x[i]);
}
if ((idx + 7) < N) {
  LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
}
```

- 把 8 个 half（正好 16 字节 = 128 位）装入局部数组；
- 借助 reinterpret 成 `float4` 的 `LDST128BITS` 宏，让一次访存指令读写完整的
  128 位，比 unpack 版本的多次 `half2` 访存更省指令，实测也是 FP16 里最快的；
- 局部数组在 PTX 中通常对应 `.local` 空间（可寻址），
  但配合 `#pragma unroll` 与编译期常数下标，编译器有机会把它优化到寄存器里，
  从而避免真正访问 local memory；
- 循环步长是 `i++`：Swish **没有成对指令**（不像 ReLU 有 `__hmax2`），
  8 个元素只能逐个 half 计算；这一点与 sigmoid 模块的 pack 版本写法相同。

## 边界约束（已识别，本次不修复）

本节记录的是**实测确认**的行为（测试环境：RTX 5070 Ti / sm_120 /
CUDA 13.0 / torch 2.14.0+cu130 / `TORCH_CUDA_ARCH_LIST=Blackwell`）。
本次只加注释、不改逻辑，因此全部保持原样。
好消息是：Swish **没有** gelu 那种"溢出成 NaN"的问题，
只有下面这两条精度 / 边界行为差异。

### 1. FP16 在 `x <= -11.09` 时饱和成 `-0.0`

实测数据（FP16 各版本行为一致）：

| 输入 `x` | 自定义 `swish_f16` | `torch.nn.functional.silu`（half） |
| --- | --- | --- |
| -5.0 | -0.033477783203125 | -0.033477783203125 |
| -11.0 | -0.00018358230590820312 | -0.0001837015151977539 |
| -12.0 | **-0.0** | -7.37309455871582e-05 |
| -88.0 | -0.0 | -0.0 |
| 11.0 | 11.0 | 11.0 |
| 88.0 | 88.0 | 88.0 |
| `inf` | `inf` | `inf` |
| `-inf` | `NaN` | `NaN` |
| `NaN` | `NaN` | `NaN` |

原因：`hexp(-x)` 在 `-x > ln(65504) ≈ 11.09` 时溢出成 `inf`，
`__hdiv(1, inf) = 0`，于是结果是 `x * 0 = -0.0`。
数学极限确实是 0，所以**绝对误差最大只有 1.8e-4**，通常无感；
但 PyTorch 把 half 提升到 float 计算 sigmoid，能给出真实的
`-7.37e-05` 这样的小值。如需让 FP16 版本在小负值区也精确，
可以改成 `x * __hdiv(1, 1 + hexp(-x))` 之外的稳定形式
（例如对 `x < -11` 单独走 `x * hexp(x)` 的近似路径）。

注意最后三行：`inf`、`-inf`、`NaN` 的行为与 PyTorch **完全一致**
（`-inf` 时 `-inf * 0` 在 IEEE 语义下就是 NaN），不属于本模块的缺陷。

### 2. FP32 的极小负数会被下溢成 `-0.0`

`swish_f32(-88)` 返回 `-0.0`，而 `x * torch.sigmoid(x)` 给出
`-5.3e-37`（一个次正规数）。原因是 `--use_fast_math` 会打开
flush-to-zero（FTZ），把次正规结果直接清成 0。
两者的绝对误差小于 1e-36，对任何实际用途都没有影响。

### 3. 向量化版本的尾部（tail）分支缺失

elementwise 的向量化版本用 `(idx + 3) < N` 判断整段元素是否齐全，
并配了尾部标量回退分支；本模块的向量化版本**两个都没有**：

| kernel | 现有判断 | `N` 不是向量宽度的整数倍时会发生什么 |
| --- | --- | --- |
| `swish_f32x4_kernel` | `idx < N`（在 load 之前） | 最后一个 block 的部分线程会把 `idx+1..idx+3` 写到 `y` 合法范围之外（越界写 1~3 个 float） |
| `swish_f16x2_kernel` | `idx < N`（在 load 之前） | 同上，越界写 1 个 half |
| `swish_f16x8_kernel` | `(idx + 0/2/4/6) < N`（在 store 处） | 同上，越界写 1 个 half |
| `swish_f16x8_pack_kernel` | `(idx + 7) < N`（在 store 处） | 方向相反：末尾不足 8 个的元素被整块**丢弃**（漏算），但不会越界 |

本模块基准测试的 `S`/`K` 都取 1024 的倍数（元素数是 256 的倍数），
启动的线程恰好整段对齐，因此不会暴露该问题。
本次只加注释、不改逻辑，所以该风险仅在本文档和源码注释中写明，
修法可参照 elementwise 模块的尾部处理。

### 4. `torch_swish` 对照实现是"两个 kernel"而不是一个

`swish.py` 里的对照实现是：

```python
def torch_swish(x, out=None):
    if out is None:
        return x * torch.sigmoid(x)
    else:
        torch.sigmoid(x, out=out)
        out.mul_(x)
        return out
```

- 数学上与 `swish(x)` 等价，但**执行上是两个 elementwise kernel**：
  先算 `sigmoid`（读写一遍显存），再 `mul_`（再读写一遍显存），
  中间还要多一次 `x` 的读取；
- 这不是"PyTorch 能做到的最快实现"，真正公平的对照应该是融合算子
  `torch.nn.functional.silu`（单个 kernel）。本次保持脚本原样不替换，
  仅在文档中说明这个差异；
- 这也解释了为什么 `kernels/swish/README.md` 里 `out_f32_th`
  比自定义 kernel 慢好几倍（`S=4096, K=4096`：0.487 ms 对 0.193 ms）。

## 启动配置（host 端）

启动端把每个 block 负责的元素数固定为 **256**：

```c
dim3 block(256 / (n_elements));   // 每个线程负责 n_elements 个元素
dim3 grid((N + 256 - 1) / 256);   // 向上取整
```

各版本的参数对应关系：

| 版本 | `n_elements` | block 线程数 | 每 block 处理元素数 |
| --- | ---: | ---: | ---: |
| `swish_f32` | 1 | 256 | 256 |
| `swish_f32x4` | 4 | 64 | 256 |
| `swish_f16` | 1 | 256 | 256 |
| `swish_f16x2` | 2 | 128 | 256 |
| `swish_f16x8` | 8 | 32 | 256 |
| `swish_f16x8_pack` | 8 | 32 | 256 |

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

`swish.cu` 通过 `TORCH_BINDING_SWISH` 宏批量生成 6 个 host 函数，
再经 `PYBIND11_MODULE` 暴露为 Python 可调用模块，主要工作：

1. 用 `CHECK_TORCH_TENSOR_DTYPE` 检查输入输出张量数据类型；
2. 根据维度/形状计算 `grid` 与 `block`；
3. 用 `reinterpret_cast<element_type *>(x.data_ptr())` 拿到裸指针；
4. 以 `kernel<<<grid, block>>>` 启动内核。

### 调用链（以 `swish_f32` 为例）

```text
swish.py
  └─ lib.swish_f32(x, y)                     # Python 调用（pybind11）
      └─ swish_f32(torch::Tensor x, y)       # C++ host 包装函数（宏生成）
          ├─ CHECK_TORCH_TENSOR_DTYPE(x/y)   # 校验 dtype 必须是 float32
          ├─ 计算 dim3 block / dim3 grid
          └─ swish_f32_kernel<<<grid, block>>>(...)   # CUDA 启动语法
              └─ cudaLaunchKernel(...)       # nvcc 生成，交给 CUDA runtime
                  └─ GPU 各线程并行执行 kernel 函数体
```

要点：

- Python 调用的 `swish_f32` 运行在 CPU 上，负责类型检查、计算 `grid` / `block`；
- `swish_f32_kernel` 运行在 GPU 上，必须用 `<<<>>>` 启动，不能普通函数调用；
- 启动后 GPU 的每个线程都会执行一遍 kernel 函数体，靠 `threadIdx` / `blockIdx`
  区分自己负责的下标 `idx`。

## 测试脚本

`swish.py` 流程：

1. 用 `torch.utils.cpp_extension.load` 现场编译 `swish.cu`；
2. 对 `S ∈ {1024, 2048, 4096}`、`K ∈ {1024, 2048, 4096}` 组合生成随机张量；
3. `run_benchmark` 先做 warmup，再运行 1000 次取平均；
4. 依次对比 6 个自定义 kernel 与 `torch_swish` 的正确性和耗时。

脚本在 `load(...)` 完成后会打印 PyTorch CUDA 扩展的实际构建目录，
便于确认 `swish_lib.so` 的存放位置；随后用
`print(torch.cuda.get_device_name())` 打印当前 GPU 设备名，
便于对照不同显卡上的耗时数据。

打印对照结果前，`run_benchmark` 会通过
`out.flatten().detach().cpu().numpy().tolist()[:2]` 把 GPU 上的结果张量
拉平、断开可能的梯度追踪（本脚本已关闭 autograd，属于保险写法）、拷回 CPU、
转成 Python 列表，并只取前 2 个元素用于人工核对；本模块还额外用
`f"{v:<12}"` 把每个数值左对齐补齐到 12 个字符，所以
`kernels/swish/README.md` 里的输出带引号和多余空格，属正常格式。

判断正确性要看同组里 6 个版本与 `out_f32_th` 是否**数值一致**：
FP32 各版本应当完全一致（含正负号与量级）；
FP16 版本因"纯 half 计算 vs PyTorch 提升到 float 计算"会有末位差异。
输入是 `torch.randn`（正负混合），所以前 2 个元素既可能是正数也可能
是 `-0.xxxx`，都属正常。

## 性能观察与结论

从 `kernels/swish/README.md` 的测试输出可总结：

- **FP16 明显快于 FP32**：`S=4096, K=4096` 时 `f32` 约 0.193 ms、
  `f16` 约 0.057 ms，约 3.4 倍；`f16x8pack` 约 0.027 ms，约 7 倍；
- **FP16 严格遵循"访存越宽越快"**：同一组里 `f16` 0.057 ms →
  `f16x2` 0.043 ms → `f16x8` 0.031 ms → `f16x8pack` 0.027 ms，
  说明 FP16 下瓶颈主要在显存带宽；
- **FP32 向量化只有小幅收益**：小尺寸下 `f32x4`（0.0101 ms）比
  `f32`（0.0125 ms）快约 20%，但大尺寸下两者持平
  （0.1926 ms 对 0.1925 ms）——此时已经撞上带宽上限，
  减少访存指令条数不再有额外收益。这与 sigmoid 模块的结论一致
  （两者都是"一次 exp + 一次除法"的计算量级）；
- **与对照实现的比例是预期结果**：`out_f32_th` 在 `S=4096, K=4096` 时
  约 0.487 ms，约是自定义 kernel 的 2.5 倍，原因是它由
  `sigmoid` + `mul_` 两个 kernel 组成、多读多写了一遍显存
  （不是与 `torch.nn.functional.silu` 的公平对比，见上文说明）；
- FP16 时官方实现（0.0597 ms）与标量版（0.0569 ms）接近，
  比 `f16x8pack`（0.0271 ms）慢一倍以上，同样是因为 PyTorch 的 half
  kernel 要先把 half 转 float 再算；
- 小规模（如 `S=1024, K=1024`）时各版本耗时差异变小甚至出现波动，
  因为 kernel 时间已经接近启动开销的量级。

## 后续可以尝试的优化方向

- 把脚本里的对照实现从 `sigmoid` + `mul_` 换成融合算子
  `torch.nn.functional.silu`，得到真正公平的性能对比；
- 补上向量化版本的尾部（tail）分支，使任意 `N` 都安全
  （顺带消除 `f16x8_pack` 的漏算问题）；
- 用 `sigmoid` 的等价稳定形式处理 `x <= -11.09` 的 FP16 小负值区，
  让饱和行为更接近 PyTorch（代价是分支或额外指令）；
- 尝试 `__expf` / `__frcp_rn` 等快速内在函数（本模块已开
  `--use_fast_math`，实际已走到快速指令路径），对比精度与速度的取舍；
- 把 Swish 的 `beta` 参数化（`swish(x) = x * sigmoid(beta * x)`，
  本模块固定 `beta = 1`），对齐原始论文的完整形式；
- 引入 grid-stride loop，使固定大小的 grid 也能处理任意大 `N`；
- 对比不同 `block` 大小、不同向量宽度对带宽的实测影响。
