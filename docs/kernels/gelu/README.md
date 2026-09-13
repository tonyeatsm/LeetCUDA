# GELU（Gaussian Error Linear Unit，高斯误差线性单元）模块设计说明

## 模块目标

本模块用于演示 CUDA 中逐元素 **GELU（高斯误差线性单元）** 的实现。
PyTorch 支持两种形式（`torch.nn.GELU` 的 `approximate` 参数）：

```text
精确式（approximate='none'）：gelu(x) = x * Φ(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
tanh 近似（approximate='tanh'）：gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

其中 `Φ(x)` 是标准正态分布的累积分布函数（CDF）。本模块的 kernel 默认走
**tanh 近似**（`GELU_OPS` / `HALF_GELU_OPS` 宏指向 `gelu_tanh_approximate`），
同时保留了 FP32 的精确式实现 `gelu_none_approximate`（用 `erff`），
只要改一行宏定义就能切换，便于对比两种近似的差异。

同时展示同一个运算的不同优化写法：

1. 标量版本：一个线程处理一个元素；
2. 向量化版本（`float4` / `half2` / 128-bit 打包）：一个线程一次处理多个连续元素；
3. FP16 版本：用更小精度的数据类型，成倍降低访存字节数。

本模块与 elementwise、sigmoid、relu、elu 模块是**姊妹示例**：6 个 kernel 版本
一一对应，区别只是把 `c[i] = a[i] + b[i]` / `y[i] = sigmoid(x[i])` /
`y[i] = max(0, x[i])` 换成 `y[i] = gelu(x[i])`。前面模块讲过的线程编号、
向量化访存、FP16 打包等通用知识本模块不再重复，下面重点说明 GELU 特有的三点：

- **公式更长**：一次 `tanh`（或 `erf`）外面还套着 `x` 的三次多项式，
  计算量比 sigmoid / elu 都大；
- **必须做溢出保护**：本模块的 tanh 是用 `exp` 拼出来的，指数必然有溢出边界，
  因此需要 `MAX_EXP_*` / `MIN_EXP_*` 这组 clamp 宏；
- **FP16 精度陷阱最深**：half 没有 `tanh`，只能用 `hexp` 拼，
  于是同时踩到"溢出成 NaN"和"相减抵消"两个坑（详见"边界约束"）。

最终目标不是追求极致性能，而是让学习者直观理解 CUDA 编程模型与访存优化思想。

## 本次改动范围

本模块参照 elementwise / sigmoid / relu 模块的做法，统一补充教学注释，
改动原则是**只加注释、不改 kernel 逻辑**，具体包含：

1. 先更新本文档（`docs/kernels/gelu/README.md`）；
2. 给 `kernels/gelu/gelu.cu` 补充分节标题、调用链、grid/block 速查、
   GELU 数学性质说明，以及逐行行内注释；
3. 给 `kernels/gelu/gelu.py` 补充模块 docstring 与行内注释，并新增
   打印 PyTorch CUDA 扩展实际构建目录、以及打印当前 GPU 设备名的代码
   （对齐 `relu.py` / `sigmoid.py`）。

`gelu.py` 的**运行行为保持不变**：仍然用 `torch.gelu = torch.nn.GELU("tanh")`
构造 tanh 近似版本的对照实现，结果打印仍保留原始的 `[...]` 列表格式，
因此 `kernels/gelu/README.md` 中的测试输出样例无需改动。

`scripts/README.md` 中补充了 gelu 模块的运行命令。

## 涉及文件

| 文件 | 作用 |
| --- | --- |
| `kernels/gelu/gelu.cu` | CUDA kernel 定义 + PyTorch C++ 绑定（代码中已添加详细注释） |
| `kernels/gelu/gelu.py` | 用 `torch.utils.cpp_extension.load` 编译并执行基准测试 |
| `kernels/gelu/README.md` | 模块使用说明、FP16 误差实验与测试输出 |

## 注释导览

对照 elementwise / sigmoid / relu 模块，`gelu.cu` 中的注释按下面的顺序组织：

| 位置 | 内容 |
| --- | --- |
| 文件头 | 模块公式（精确式 vs tanh 近似）、6 个版本概览、学习重点 |
| 【GELU 曲线速览】 | ASCII 折线图、采样值、负向小凹陷 / 非单调 / 与 ReLU、SiLU 的对照 |
| 宏定义段 | 访存宏、`MAX_EXP_*` / `MIN_EXP_*` 的来源与"夹取位置有坑"的说明、`GELU_OPS` 切换方式 |
| `gelu_tanh_approximate` | tanh 的 `exp` 实现、FP16 相减抵消误差、`gelu_none_approximate` 精确式 |
| `gelu_f32_kernel` | 调用链、grid/block 速查、clamp + tanh 的计算顺序、逐行注释 |
| `gelu_f32x4_kernel` | `float4` 向量化访存、先无条件 load 再判断 store 的边界风险 |
| `gelu_f16_kernel` | FP16 常数转换开销、`__hmin` / `__hmax` 选型 |
| `gelu_f16x2_kernel` | `half2` 双分量计算 |
| `gelu_f16x8_kernel` | unpack 写法：4 个 `half2` 完成 8 个元素 |
| `gelu_f16x8_pack_kernel` | pack 写法：局部数组 + 128 位整体访存 |
| `TORCH_BINDING_GELU` | host 端宏展开细节与调用链 |
| `PYBIND11_MODULE` | 导出给 Python 的函数名 |

`TORCH_BINDING_GELU` 宏内部的注释使用 `/* ... */` 块注释而非 `//`：
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

### 2. GELU 的数学性质

```text
gelu(x) = x * Φ(x)，Φ 为标准正态 CDF
        = 0.5 * x * (1 + erf(x / sqrt(2)))                                    （精确式）
       ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))             （tanh 近似）
```

函数曲线（ASCII 示意图，`x ∈ [-3, 3]`，tanh 近似）：

```text
 4.0 ┤                          ***
     ┤                       ***
 3.0 ┤                    ***
     ┤                 ***
 2.0 ┤              ***
     ┤           ***
 1.0 ┤        ***
 0.0 ┤*******+                     ← 唯一的零点在 x = 0
     ┤    ***
-0.17┤   *                        ← 负向小凹陷，min ≈ -0.1700（x ≈ -0.752）
     └──────────────────────────────→ x
     -3   -2   -1   0   1   2   3
```

采样值（tanh 近似）：`-3 → -0.0036`、`-2 → -0.0454`、`-1 → -0.1588`、
`-0.75 → -0.1700`、`0 → 0`、`1 → 0.8412`、`2 → 1.9546`、`3 → 2.9964`。

读图要点：

- **正半轴近似恒等映射**：`x` 稍大（约 3 以上）后 `Φ(x) ≈ 1`，
  `gelu(x) ≈ x`，这一点和 ReLU / ELU 的正半轴一致；
- **负半轴不是硬零**：`x < 0` 时输出是小负数，随 `x` 减小先变负、再回到 0，
  最小值约 `-0.1700`（出现在 `x ≈ -0.752`），所以 GELU **不是单调函数**；
- **在 `x = 0` 处光滑可导**（曲线整体是无穷阶可导的），
  既没有 ReLU 的"0 点不可导"，也没有 ELU 的"负半轴带一个转折"；
- **负半轴梯度不为 0**：和 ELU 一样避免了死亡 ReLU 问题，
  但它的负值区间更浅（最深 -0.17，ELU 可以一路滑到 -1）；
- **计算量最大**：一次 tanh（内部还要一次 exp）加若干次乘加，
  在 relu / elu / sigmoid / gelu / swish 这一族里属于计算偏重的算子，
  因此它的优化重点往往不在访存，而在"少算一点"（换近似、降精度，
  或者干脆换成 swish 这种更便宜的平滑激活）。

与同族算子的对照（下文会反复引用这张表）：

| 对比项 | relu | elu（alpha = 1） | gelu（tanh 近似） | swish / silu |
| --- | --- | --- | --- | --- |
| 公式 | `max(0, x)` | `x > 0 ? x : exp(x) - 1` | `0.5x(1 + tanh(k(x + 0.044715x^3)))`，`k = sqrt(2/pi)` | `x * sigmoid(x)` |
| 值域 | `[0, +∞)` | `(-1, +∞)` | `(-0.17, +∞)` | `(-0.278, +∞)` |
| 单调性 | 单调 | 单调 | 非单调（负半轴有凹陷） | 非单调（负半轴有凹陷） |
| `x = 0` 处 | 不可导 | 可导，导数 1 | 光滑可导 | 光滑可导 |
| 负半轴梯度 | 恒为 0 | `exp(x) > 0` | 不为 0 | 不为 0 |
| 每元素计算 | 一次取最大值 | 一次比较 + 一次 `exp` | 多次乘加 + 一次 tanh（内含 `exp`） | 一次 `exp` + 加法 + 除法 + 乘法 |
| 是否需要 clamp | 不需要 | 不需要 | **需要**（见下节） | 不需要 |

### 3. 为什么 GELU 需要 clamp，以及 clamp 该夹谁

本模块用下面的等价形式实现 tanh：

```text
tanh(t) = (exp(2t) - 1) / (exp(2t) + 1)
```

这就把问题转移到了 `exp` 上：只要 `2t` 超过 `exp` 的溢出阈值，
`exp(2t)` 就变成 `inf`，于是 `(inf - 1) / (inf + 1) = inf / inf = NaN`。
所以 GELU 必须像 sigmoid 一样先把参数夹进安全区间。

边界宏的含义（与 `docs/kernels/sigmoid/README.md` 同源）：

| 宏 | 取值 | 来源 |
| --- | --- | --- |
| `MAX_EXP_F32` | `88.3762626647949f` | 逐元素 sigmoid 常用的 FP32 保守上界（略小于 `ln(FLT_MAX) ≈ 88.723`） |
| `MIN_EXP_F32` | `-88.3762626647949f` | 与上界对称取负 |
| `MAX_EXP_F16` | `__float2half(11.089866488461016f)` | `ln(65504) ≈ 11.09`，65504 是 half 的最大规格数 |
| `MIN_EXP_F16` | `__float2half(-9.704060527839234f)` | `ln(2^-14) ≈ -9.70`，`2^-14` 是 half 的最小规格正数 |

FP16 的上下界**不对称**，是因为 half 的指数范围本身就偏向 0 一侧：
最大只能到 65504（`ln ≈ 11.09`），最小规格正数却是 `2^-14 ≈ 6.1e-5`
（`ln ≈ -9.70`）。

**关键细节（本模块的核心陷阱）**：代码夹的是输入 `x`，但送进 `exp` 的是
`2 * inner`，其中

```text
inner = sqrt(2/pi) * (x + 0.044715 * x^3) ≈ 0.797885 * (x + 0.044715 * x^3)
```

三次项让 `inner` 比 `x` 增长得快得多：`x = 4.0` 时 `2 * inner ≈ 11.05`
（刚好没溢出），`x = 4.1` 时就已经超过 11.09 而溢出。
所以 FP16 版本在 `x ≳ 4.03` 时会输出 **NaN**——这是本模块已识别的问题，
详见下文"边界约束"。FP32 的边界（88.376）远大于三次项造成危险的范围，
因此 FP32 版本没有这个问题。

顺带一提：`x` 很大时 GELU 本身已经饱和（`gelu(x) = x`），
所以 clamp 掉尾部理论上不改变结果——前提是夹住"真正送进 `exp` 的那个量"。

### 4. FP16 精度提醒

FP32 尾数 23 位，最小刻度约 `2^-23 ≈ 0.00000012`（约 7 位有效数字）；
FP16 尾数只有 10 位，最小刻度为 `2^-10 = 1/1024` ≈ `0.00098`
（约 3~4 位有效数字，保守按约 3 位）。

GELU 在 FP16 下有**三处**需要额外留意的地方：

1. **`hexp` 拼 tanh 的相减抵消**：`inner` 接近 0 时 `hexp(2 * inner) ≈ 1`，
   `(1 + δ - 1) / (1 + δ + 1)` 的分子是"两个接近的数相减"，
   有效位数大量丢失。这是 `kernels/gelu/README.md` 里"half 误差实验"
   想说明的现象：把 half 提升成 float 再算 tanh，结果就和 PyTorch 一致了；
2. **PyTorch 的 half GELU 是在 float 里算的**：PyTorch 对 half 输入会先把
   数据类型提升到 float 做完 tanh / erf 再转回 half，所以它的结果比本模块
   纯 half 计算更准。两者通常只差最后 1~2 位十进制数字，
   例如 `-0.06713867`（kernel）对 `-0.06707764`（torch）；
3. **溢出成 NaN**：见上文，`x ≳ 4.03` 时 FP16 版本输出 NaN。

另外，FP16 的最大规格数只有 65504，`torch.randn` 生成的输入远小于它，
不会触发输入本身的上溢。

## Kernel 设计

所有 kernel 都遵守一个约定：**把输入输出都当作长度为 `N` 的一维数组处理**。
二维矩阵 `(S, K)` 在启动端被展平为 `N = S * K`。

### 0. 系数宏与三个辅助函数

```c
#define SQRT_2_PI M_SQRT2 *M_2_SQRTPI * 0.5f          // ≈ 0.7978845608 = sqrt(2/pi)
#define HALF_SQRT_2_PI __float2half(M_SQRT2) * __float2half(M_2_SQRTPI) * HALF_DIV2
#define HALF_V_APP __float2half(0.044715f)
#define HALF_GELU_OPS gelu_tanh_approximate
#define GELU_OPS gelu_tanh_approximate
```

- `M_SQRT2 = sqrt(2) ≈ 1.41421356`、`M_2_SQRTPI = 2/sqrt(pi) ≈ 1.12837917`，
  两者相乘再乘 `0.5` 正好是 `sqrt(2/pi) ≈ 0.79788456`，
  也就是 tanh 近似公式里最前面那个系数；
- 源码注释把该式写成 `sqrt(2 * pi) / 2`，但这个式子实际等于
  `sqrt(pi/2) ≈ 1.2533`，与代码算出的 `0.79788` 不是同一个值
  （正确的等价写法是 `2 / sqrt(2 * pi)`）。**代码的取值是正确的**，
  本次保留原注释，只在旁边补一句说明；
- 宏 `SQRT_2_PI` **没有加括号**，靠"乘法同优先级、从左到右"恰好算对。
  一旦它出现在优先级更高的表达式里（例如紧跟除号），就会静默出错，
  规范写法是 `#define SQRT_2_PI ((M_SQRT2) * (M_2_SQRTPI) * 0.5f)`（本次不改）；
- `HALF_SQRT_2_PI` 用"先由 FP32 字面量转出 half、再做 half 乘法"的方式
  构造 half 系数，与 `HALF_DIV2 = __float2half(0.5f)` 配合；
- `GELU_OPS` / `HALF_GELU_OPS` 是**算法选择开关**：默认指向 tanh 近似，
  把 `GELU_OPS` 改成 `gelu_none_approximate` 就切换成 FP32 精确式
  （`erff` 版本），`kernels/gelu/README.md` 里演示的正是这类对照实验。

三个辅助函数对应三种精度 / 算法组合：

| 函数 | 精度 | 算法 | 说明 |
| --- | --- | --- | --- |
| `gelu_tanh_approximate(float)` | FP32 | tanh 近似 | 直接调用 `tanhf` |
| `gelu_tanh_approximate(half)` | FP16 | tanh 近似 | 用 `hexp` 拼出 tanh |
| `gelu_none_approximate(float)` | FP32 | 精确式 | 用 `erff`（`GELU_OPS` 可切到它） |

FP16 版 tanh 的实现是：

```c
half x_cube = x * x * x;
half inner = HALF_SQRT_2_PI * (x + HALF_V_APP * x_cube);
return HALF_DIV2 * x *
       (HALF_1 + ((hexp(inner * HALF_2) - HALF_1) / (hexp(inner * HALF_2) + HALF_1)));
```

注意 `inner * HALF_2` 在表达式里出现了两次，编译器一般会用公共子表达式消除
（CSE）复用结果，但指数本身在**表达式层面**是重复的——也正是在这里，
`hexp` 一旦溢出就会产生 `inf / inf = NaN`。

### 1. `gelu_f32_kernel`（FP32 标量版）

```c
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N) {
  float v = fminf(fmaxf(x[idx], MIN_EXP_F32), MAX_EXP_F32);
  y[idx] = GELU_OPS(v);
}
```

- 一个线程只算一个元素；
- `if (idx < N)` 处理 `N` 不能被 block 大小整除时"多启动"的线程；
- `fminf` / `fmaxf` 是 float 版本（写成 `fmin` / `fmax` 会提升成 double，更慢）；
- 顺序是"先夹取、再算 GELU"，不能反：夹取是为了让后续 tanh 保持有限值；
- `tanhf` 是 CUDA 的 float 双曲正切函数；`--use_fast_math` 会让它走到
  更快但精度略低的实现路径（本模块默认开启，与其它模块保持一致）。

### 2. `gelu_f32x4_kernel`（FP32 向量化版）

每个线程连续处理 **4 个 float（16 字节）**：

```c
int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
float4 reg_x = FLOAT4(x[idx]);       // 无条件 load
float4 reg_y;
reg_x.x = fminf(fmaxf(reg_x.x, MIN_EXP_F32), MAX_EXP_F32);   // 4 个分量逐个 clamp
...
reg_y.x = GELU_OPS(reg_x.x);                                  // 4 个分量逐个算 GELU
...
if ((idx + 0) < N) {
  FLOAT4(y[idx]) = reg_y;            // 只在段首合法时 store
}
```

- 一次 16 字节的向量 load/store 代替 4 次 4 字节访存，减少访存指令条数；
- 与 elementwise 不同，GELU 是**单输入**算子：只读 `x` 一份数据；
- 注意 `float4 reg_x = FLOAT4(x[idx]);` **在越界判断之前无条件执行**，
  只有后面的 store 受 `(idx + 0) < N` 保护。当 `N` 不是 4 的倍数时，
  最后一组线程会读到 `x` 末尾之外（越界读），写回则被拦住；
- GELU 计算偏重，向量化省下的访存指令在整个 kernel 里占比不高，
  所以实测 `f32x4` 往往**不比标量版快**（见文末"性能观察"）。

### 3. `gelu_f16_kernel`（FP16 标量版）

```c
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N) {
  half v = x[idx];
  v = __hmin(__hmax(v, MIN_EXP_F16), MAX_EXP_F16);
  y[idx] = HALF_GELU_OPS(v);
}
```

- 数据类型为 `half`（2 字节），字节数减半，是后续 FP16 向量化的基础；
- 绝对值保护用 half 版本的 `__hmin` / `__hmax`（直接比较 16 位 half，
  不走 float，避免每次计算都插入类型转换）；
- 计算全部落在 `HALF_GELU_OPS` 指向的 `gelu_tanh_approximate(half)` 上。

### 4. `gelu_f16x2_kernel`（FP16 每线程 2 元素）

- 一次用 `half2`（2 个 half = 4 字节）完成两个元素的搬运；
- `half2` 只有 `x`、`y` 两个分量，两个分量各自独立 clamp 与计算；
- 与 `f32x4` 一样，`HALF2(x[idx])` 是无条件 load，store 才判 `(idx + 0) < N`，
  所以 `N` 为奇数时存在越界读。

### 5. `gelu_f16x8_kernel`（FP16 每线程 8 元素，unpack 写法）

- 每个线程处理 8 个 half，拆成 4 个 `half2`（`idx+0/2/4/6`）分别处理；
- 这里的 "unpack" 指：数据在内存里本就是连续的 8 个 half，
  本 kernel 不把它整体打包搬运，而是按 `half2` 逐段读、逐段写；
- 相比 f16x2，每个线程做更多工作，能摊薄索引计算等固定开销；
- 代码先对 8 个分量逐个 clamp，再逐个算 GELU，最后逐个 `half2` 写回；
- 4 次 `HALF2(x[...])` 读取是**无条件执行**的，越界判断只出现在写回处；
- 这里有个排版上的小细节：计算时把结果又写回了 `reg_x_*`（复用了输入寄存器），
  而不是像其它模块那样单独用 `reg_y_*` 接收结果——
  本次只补充注释，不改逻辑。

### 6. `gelu_f16x8_pack_kernel`（128-bit 打包版）

```c
half pack_x[8], pack_y[8]; // 8x16 bits=128 bits.
LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); // load 128 bits
#pragma unroll
for (int i = 0; i < 8; ++i) {
  half v = __hmin(__hmax(pack_x[i], MIN_EXP_F16), MAX_EXP_F16);
  pack_y[i] = HALF_GELU_OPS(v);
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
- 循环步长是 `i++`：GELU **没有成对指令**（没有 `htanh2` 之类），
  8 个元素只能逐个 half 计算。

## 边界约束（已识别，本次不修复）

本节记录的是**实测确认**的行为差异（测试环境：RTX 5070 Ti / sm_120 /
CUDA 13.0 / torch 2.14.0+cu130 / `TORCH_CUDA_ARCH_LIST=Blackwell`）。
本次只加注释、不改逻辑，因此全部保持原样。

### 1. FP16 版本在 `x ≳ 4.03` 时输出 NaN（最严重）

实测数据（FP16 各版本行为一致）：

| 输入 `x` | 自定义 `gelu_f16` | `torch.nn.GELU("tanh")`（half） |
| --- | --- | --- |
| 3.0 | 2.99609375 | 2.99609375 |
| 4.0 | 4.0 | 4.0 |
| 4.1 | **NaN** | 4.1015625 |
| 5.0 | **NaN** | 5.0 |
| 12.0 | **NaN** | 12.0 |
| -12.0 | -0.0 | -0.0 |

原因见"为什么 GELU 需要 clamp"一节：clamp 夹的是 `x`，但送进 `hexp` 的是
`2 * inner ≈ 1.59577 * (x + 0.044715 * x^3)`；反解 `2 * inner = ln(65504)`
得到理论阈值 **x ≈ 4.0278**，与实测（4.0 正常、4.1 已 NaN）吻合。

**影响范围（实测）**：`gelu.py` 的基准输入是 `torch.randn`
（`1 - Φ(4.03) ≈ 2.2e-5`），用 `torch.manual_seed(0)` 固定随机种子后统计：

| 形状 | 元素数 | `x > 4.03` 的元素 | `gelu_f16` 输出 NaN | torch 参考 NaN |
| --- | ---: | ---: | ---: | ---: |
| 1024×1024 | 104.9 万 | 37 | **37** | 0 |
| 4096×4096 | 1677.7 万 | 455 | **455** | 0 |

也就是说 NaN 的个数与"超过阈值的元素个数"完全一一对应。
所以 `kernels/gelu/README.md` 里 FP16 一栏的耗时数据仍可比，
但**结果里实际混有 NaN**（千分之一量级以上的比例不存在，
但绝对数量有几百个）；只打印前 2 个元素的抽查方式看不出来。

**可选修法**（本次不做，留作练习）：

- 把 clamp 换成对 `inner` 的 clamp（先算 `inner`，再夹到
  `[-9.704, 11.0899] / 2` 的范围），从根上保证 `hexp` 不溢出；
- 或者像 `kernels/gelu/README.md` 的对照实验那样，把 half 提升成 float
  再算 tanh（`__half2float` → `tanhf` → `__float2half`），代价是转换指令；
- 或者对 `|x| > 4.03` 的元素直接走饱和分支（返回 `x` 或 0）。

### 2. FP32 版本对大输入会截断，且会吞掉 NaN

实测：

| 输入 `x` | `gelu_f32` | `torch.nn.GELU("tanh")` |
| --- | --- | --- |
| 88.0 | 88.0 | 88.0 |
| 89.0 | 88.37625885009766 | 89.0 |
| 100.0 | 88.37625885009766 | 100.0 |
| `inf` | 88.37625885009766 | `inf` |
| `NaN` | 0.0（或 -0.0） | `NaN` |

两个原因都出在 clamp 这一步：

- `fminf(x, MAX_EXP_F32)` 会把所有大于 88.376 的输入截断到 88.376，
  于是 `gelu_f32(100)` 返回 88.376 而不是 100，无穷大也被截断成有限值；
- `fmaxf(NaN, MIN_EXP_F32)` 遵循 IEEE 754 的 `maxNum` 语义，
  **一个参数是 NaN 时返回另一个参数**，于是 NaN 先被换成 -88.376，
  再被 tanh 饱和成 `0.5 * (-88.376) * (1 - 1) = 0`——
  NaN 被"洗掉"了，而 PyTorch 会把 NaN 传播到输出。

本模块基准测试用 `torch.randn`（几乎都落在 ±5 内），不会触发这两条；
但作为通用算子复用时需要补上，例如把 NaN 单独判断后直接返回，
或者只在"确实要送进 `exp`"的路径上做 clamp。

### 3. FP16 的 tanh 实现精度低于 PyTorch

这是 `kernels/gelu/README.md` 里已经记录过的老问题：half 没有 `tanh`，
本模块用 `hexp` 拼出来，`inner ≈ 0` 附近存在相减抵消，
与 PyTorch（内部提升到 float 计算）相比会有 1~2 位十进制数字的偏差。
示例：同一输入下 `out_f16x8` 给出 `-0.08251953`，`out_f16_th` 给出
`-0.08197021`。数学上两者都在 half 的精度范围内，属于"实现路径不同"，
不是 bug。

### 4. 向量化版本的尾部（tail）分支缺失

elementwise 的向量化版本用 `(idx + 3) < N` 判断整段元素是否齐全，
并配了尾部标量回退分支；本模块的向量化版本**两个都没有**：

| kernel | 现有判断 | `N` 不是向量宽度的整数倍时会发生什么 |
| --- | --- | --- |
| `gelu_f32x4_kernel` | 先无条件 `FLOAT4` load，再 `(idx + 0) < N` store | 越界读 1~3 个 float；若 `idx < N` 仍成立则还越界写 1~3 个 float |
| `gelu_f16x2_kernel` | 先无条件 `HALF2` load，再 `(idx + 0) < N` store | 同上，宽度为 2 |
| `gelu_f16x8_kernel` | 无条件 4 次 `HALF2` load，store 处判 `(idx + 0/2/4/6) < N` | 越界读；写回时把尾部 1 个 half 写到 `y` 范围之外 |
| `gelu_f16x8_pack_kernel` | `(idx + 7) < N` | 方向相反：末尾不足 8 个的元素被整块**丢弃**（漏算），但不会越界 |

本模块基准测试的 `S`/`K` 都取 1024 的倍数（元素数是 256 的倍数），
启动的线程恰好整段对齐，因此不会暴露该问题。上面的 NaN 实测里
`f16x8_pack` 最后一组返回了未写入的哨兵值，就是这条的表现。

## 启动配置（host 端）

启动端把每个 block 负责的元素数固定为 **256**：

```c
dim3 block(256 / (n_elements));   // 每个线程负责 n_elements 个元素
dim3 grid((N + 256 - 1) / 256);   // 向上取整
```

各版本的参数对应关系：

| 版本 | `n_elements` | block 线程数 | 每 block 处理元素数 |
| --- | ---: | ---: | ---: |
| `gelu_f32` | 1 | 256 | 256 |
| `gelu_f32x4` | 4 | 64 | 256 |
| `gelu_f16` | 1 | 256 | 256 |
| `gelu_f16x2` | 2 | 128 | 256 |
| `gelu_f16x8` | 8 | 32 | 256 |
| `gelu_f16x8_pack` | 8 | 32 | 256 |

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

`gelu.cu` 通过 `TORCH_BINDING_GELU` 宏批量生成 6 个 host 函数，
再经 `PYBIND11_MODULE` 暴露为 Python 可调用模块，主要工作：

1. 用 `CHECK_TORCH_TENSOR_DTYPE` 检查输入输出张量数据类型；
2. 根据维度/形状计算 `grid` 与 `block`；
3. 用 `reinterpret_cast<element_type *>(x.data_ptr())` 拿到裸指针；
4. 以 `kernel<<<grid, block>>>` 启动内核。

### 调用链（以 `gelu_f32` 为例）

```text
gelu.py
  └─ lib.gelu_f32(x, y)                      # Python 调用（pybind11）
      └─ gelu_f32(torch::Tensor x, y)        # C++ host 包装函数（宏生成）
          ├─ CHECK_TORCH_TENSOR_DTYPE(x/y)   # 校验 dtype 必须是 float32
          ├─ 计算 dim3 block / dim3 grid
          └─ gelu_f32_kernel<<<grid, block>>>(...)   # CUDA 启动语法
              └─ cudaLaunchKernel(...)       # nvcc 生成，交给 CUDA runtime
                  └─ GPU 各线程并行执行 kernel 函数体
```

要点：

- Python 调用的 `gelu_f32` 运行在 CPU 上，负责类型检查、计算 `grid` / `block`；
- `gelu_f32_kernel` 运行在 GPU 上，必须用 `<<<>>>` 启动，不能普通函数调用；
- 启动后 GPU 的每个线程都会执行一遍 kernel 函数体，靠 `threadIdx` / `blockIdx`
  区分自己负责的下标 `idx`。

## 测试脚本

`gelu.py` 流程：

1. 用 `torch.utils.cpp_extension.load` 现场编译 `gelu.cu`；
2. 对 `S ∈ {1024, 2048, 4096}`、`K ∈ {1024, 2048, 4096}` 组合生成随机张量；
3. `run_benchmark` 先做 warmup，再运行 1000 次取平均；
4. 依次对比 6 个自定义 kernel 与 PyTorch 官方 tanh 近似 GELU 的正确性和耗时。

脚本在 `load(...)` 完成后会打印 PyTorch CUDA 扩展的实际构建目录，
便于确认 `gelu_lib.so` 的存放位置；随后用
`print(torch.cuda.get_device_name())` 打印当前 GPU 设备名，
便于对照不同显卡上的耗时数据。

对照实现对初学者来说有点绕，这里说明一下原始写法：

```python
torch.gelu = torch.nn.GELU("tanh")
...
run_benchmark(partial(torch.gelu), x, "f32_th")
```

- `torch.nn.GELU("tanh")` 构造的是一个 **nn.Module 实例**（默认参数是
  `approximate='none'`，这里显式选 `'tanh'`，正好与本模块 kernel 的
  tanh 近似对齐；如果漏掉这个参数，对照结果会变成 erf 精确式，
  两者在小数点后第 4 位左右就会出现差异）；
- 该模块实例被赋给 `torch.gelu` 这个名字（`torch` 命名空间里原本没有
  `gelu`，`torch.nn.functional.gelu` 才是有名有姓的那个 API），
  所以这里是**覆盖式的临时写法**，只为让下面的调用看起来像 `torch.gelu(x)`；
- 用 `partial(...)` 包一层是为了得到"可调用对象"，
  与其它模块里 `partial(torch.sigmoid, out=y)` 的用法保持一致；
- 因为对照走的是模块实例，脚本里**没有传 `out=`**，
  `run_benchmark` 会自动走"不传 out"的分支（返回新张量）。

打印对照结果前，`run_benchmark` 会通过
`out.flatten().detach().cpu().numpy().tolist()[:2]` 把 GPU 上的结果张量
拉平、断开可能的梯度追踪（本脚本已关闭 autograd，属于保险写法）、拷回 CPU、
转成 Python 列表，并只取前 2 个元素用于人工核对。
本模块**没有**做 `f"{v:<12}"` 左对齐补齐，所以
`kernels/gelu/README.md` 里的输出是干净的 `[-0.13358943, -0.06881647]` 形式。

判断正确性要看同组里 6 个版本与 `out_fXX_th` 是否**数值一致**：
FP32 各版本应当完全一致；FP16 版本因"纯 half 计算 vs PyTorch 提升到 float
计算"会有末位差异；同时请留意上文提到的 NaN 问题——
抽查前 2 个元素是发现不了的。

## 性能观察与结论

从 `kernels/gelu/README.md` 的测试输出可总结：

- **FP16 明显快于 FP32**：`S=4096, K=4096` 时 `f32` 约 0.191 ms、
  `f16` 约 0.055 ms，约 3.5 倍；`f16x8pack` 约 0.027 ms，约 7 倍；
- **FP32 向量化基本没有收益甚至变慢**：同一组里 `f32x4`（0.204 ms）
  反而略慢于 `f32`（0.191 ms）。原因是 GELU 的算术强度高
  （一次 tanh 内部就含一次 exp），访存指令数不是瓶颈，
  向量化那点收益被"逐分量展开的冗长计算"吃掉了；
- **FP16 严格遵循"访存越宽越快"**：`f16` 0.055 ms → `f16x2` 0.036 ms →
  `f16x8` 0.031 ms → `f16x8pack` 0.027 ms，说明 FP16 下瓶颈回到了显存带宽；
- **与 PyTorch 官方算子的对比**：
  - FP32 时官方算子通常更快（`S=1024, K=4096`：0.0093 ms 对 0.0427 ms），
    PyTorch 的融合 kernel 用 float 计算并做了更细的向量化；
  - 大尺寸时两者接近（`S=4096, K=4096`：0.205 ms 对 0.191 ms 反超）；
  - FP16 时本模块明显更快（同一组：0.0389 ms 对 0.0274 ms），
    因为 PyTorch 的 half GELU 要先把 half 转 float 算完再转回 half，
    多出转换开销，而本模块是纯 half 计算（代价就是精度更低、且会 NaN）；
- 小规模（如 `S=1024, K=1024`）时各版本耗时差异变小甚至出现波动，
  因为 kernel 时间已经接近启动开销的量级。

## 后续可以尝试的优化方向

- **优先修 FP16 的 NaN 问题**（详见"边界约束"节），这是本模块唯一会
  产生错误数值（而不只是精度下降）的问题；
- 补上向量化版本的尾部（tail）分支，使任意 `N` 都安全
  （顺带消除 `f16x8_pack` 的漏算问题）；
- 把 clamp 改成"只在要送进 `exp` 的路径上钳位"，同时让 NaN 正常传播，
  避免大输入被截断到 88.376；
- 给 `SQRT_2_PI` 宏补上括号，避免宏展开的优先级隐患；
- `gelu.cu` 已经 `#include <cuda_bf16.h>` 与 `<cuda_fp8.h>`
  （以及现成的 `BFLOAT2` 宏），但没有任何 bf16 kernel；
  可以照葫芦画瓢补一组 bf16 版本，对比 bf16 与 fp16 的精度 / 性能；
- 引入 grid-stride loop，使固定大小的 grid 也能处理任意大 `N`；
- 对比 erf 精确式与 tanh 近似在同一精度下的耗时差异，
  以及它们相对 `torch.nn.GELU("none")` / `("tanh")` 的偏差范围；
- 对比不同 `block` 大小、不同向量宽度对带宽的实测影响。
