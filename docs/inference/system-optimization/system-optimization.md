> 基于论文 *Running VLAs at Real-time Speed*（arXiv:2510.26742）
代码：https://github.com/Dexmal/realtime-vla
> 

---

## 一、问题背景与目标

### 1.1 为什么 VLA 推理慢？

π₀ 模型共有约 **33 亿参数**，由两部分组成：

| 模块 | 基础模型 | 参数量 | 特性 |
| --- | --- | --- | --- |
| VLM（视觉语言模型） | PaliGemma（SigLIP + Gemma） | ~3B | 计算密集（compute-bound） |
| AE（动作专家） | 精简版 Gemma | ~300M | 带宽密集（bandwidth-bound） |

朴素 PyTorch 推理需要 **106.5 ms**（2 视图），只能运行在约 9.4 Hz，远低于相机帧率 30 FPS 的要求。根本原因有三：

1. **CPU overhead**：每次推理要启动 >1000 个 CUDA kernel，每次 kernel launch 约 5 μs，累积 ~5ms 的纯调度开销
2. **冗余计算**：计算图中存在大量可合并的线性操作，产生不必要的中间内存读写
3. **kernel 效率低**：默认 cuBLAS 配置未针对模型中特定的矩阵尺寸优化，Tile 分配不均

### 1.2 目标

```
目标：< 33ms（即 30 FPS，相机全帧处理）
基线：106.5ms（naive PyTorch，2视图，RTX 4090）
```

33 ms 是关键阈值。哪怕达到 34 ms，在连续运行时也必须偶尔丢帧；若关键事件恰好发生在被丢掉的帧上，端到端延迟将增加整整一帧的时间。

---

## 二、优化总览与效果

### 2.1 逐步优化效果（RTX 4090，2 视图）

```
Naive PyTorch:      106.5 ms   ← 出发点
+ CUDA Graph:        43.5 ms   ↓ 63.0 ms，单步最大收益，消除 CPU 调度开销
+ 计算图简化:        35.7 ms   ↓  7.8 ms，折叠冗余线性层，减少 kernel 数量
+ Triton 内核优化:   27.3 ms   ↓  8.4 ms，调优分块、融合标量操作
─────────────────────────────
理论下界（Roofline）: 20.6 ms  ← 距极限仅 30%，已接近最优
```

### 2.2 多视图对比（最终优化后）

| 方法 | 1 视图 | 2 视图 | 3 视图 |
| --- | --- | --- | --- |
| Naive PyTorch | 105.0 ms | 106.5 ms | 113.9 ms |
| openpi/jax（官方） | 43.8 ms | 53.7 ms | 67.6 ms |
| **本文方法** | **20.0 ms** | **27.3 ms** | **36.8 ms** |
| 理论下界 | 13.7 ms | 20.6 ms | 27.6 ms |

> 本文方法比官方 JAX 实现快约 **2 倍**，比 naive PyTorch 快约 **4 倍**。
> 

### 2.3 我的 RTX 3080 实测

```
RTX 3080：74.5 ms（约是 RTX 4090 的 2.73 倍）

硬件差距原因：
  CUDA Cores:  16384 (4090) vs 8704 (3080)
  显存带宽:    1008 GB/s vs 760 GB/s
  L2 Cache:    72 MB vs 5 MB  ← 差距最显著，直接影响矩阵权重缓存命中率
```

---

## 三、优化一：消除 CPU 开销（CUDA Graph）

### 3.1 问题根因

神经网络推理由 Python 逐一调用 CUDA kernel，每次 launch 约有 5 μs 的 CPU 端调度开销：

```
CPU 端流程：准备参数(1μs) → CUDA Driver 处理(2μs) → 提交 GPU 队列(1μs) ≈ 5μs/次
1000 个 kernel × 5μs = 5 ms 纯 overhead（还不含 Python 解释器本身的开销）
```

CPU 与 GPU 之间存在流水线气泡：

```
不用 CUDA Graph：
CPU: |L|↓|L|↓|L|↓|L|↓|...  ← 密集的 Launch 调用，有等待
GPU:   █ █ █ █ █ █ █ █ ...  ← kernel 间有 gap

使用 CUDA Graph：
CPU: |GraphLaunch|__________  ← 只有一次 API 调用
GPU:  ████████████████████   ← kernel 连续执行，无 gap
```

### 3.2 CUDA Graph 原理

CUDA Graph 分两个阶段：

**录制阶段（只做一次）**：将一次完整推理的所有 kernel 调用序列录制为一张"图"，记录拓扑关系和所有参数指针，但不实际执行计算。

**回放阶段（每次推理）**：一次 `cudaGraphLaunch` 调用，GPU 和驱动程序直接执行整张图，彻底绕过 Python 和 CUDA Driver 的逐 kernel 调度逻辑。

**关键约束**：所有 kernel 代码和缓冲区指针必须在每次运行时保持不变（即无动态 shape、无 if/while 分支）。π₀ 模型的 Transformer 结构满足此条件。

### 3.3 代码实现（pi0_infer.py:1302）

```python
class Pi0Inference:
    def __init__(self, weights, num_views, chunk_size):
        # 1. 创建静态 buffer（地址固定，Graph 录制时绑定的就是这些地址）
        self.buffers = {
            'observation_images': torch.zeros(
                (num_views, 3, 224, 224), dtype=torch.bfloat16, device='cuda'
            ),
            'diffusion_noise': torch.zeros(
                (chunk_size, 32), dtype=torch.bfloat16, device='cuda'
            ),
            'action_output': torch.zeros(
                (chunk_size, 32), dtype=torch.bfloat16, device='cuda'
            ),
        }

        # 2. 预热（CUDA Graph 录制前需要先运行一次，让 PyTorch 完成内存分配）
        for _ in range(3):
            pi0_model(weights, self.buffers, num_views)
        torch.cuda.synchronize()

        # 3. 录制（只做一次）
        self.infer_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.infer_graph):
            pi0_model(weights, self.buffers, num_views)

    def forward(self, images, noise):
        # 4. 每次推理：拷贝输入 → 回放 → 读取输出
        self.buffers['observation_images'].copy_(images)
        self.buffers['diffusion_noise'].copy_(noise)
        self.infer_graph.replay()          # 零 CPU overhead
        return self.buffers['action_output'].clone()
```

### 3.4 与 torch.compile 的关系

代码中同时使用了 `@torch.compile`（pi0_infer.py:588, 819）：

```python
@torch.compile
def AttnMultiKey(QKV): ...

@torch.compile
def MultiAttention(query, key, value): ...
```

两者的能力对比：

|  | `torch.compile` | CUDA Graph |
| --- | --- | --- |
| 层级 | 软件层（Python → CUDA） | 驱动层（CUDA API） |
| 做什么 | 算子融合、代码生成 | 消除 kernel 启动 overhead |
| 消除 Python overhead | ✅ 90%+ | ✅ 95%+ |
| 算子融合 | ✅ 自动 | ❌ |
| 支持动态 shape | ✅ | ❌ |

两者在"消除 Python overhead"上有约 50% 的功能重叠。正因为代码中已有 `@torch.compile`，CUDA Graph 在 π₀ 模型上额外收益只有约 **3%**：

```
只用 torch.compile：  ~45 ms
只用 CUDA Graph：     ~50-60 ms
两者结合：            ~43.5 ms（额外收益仅 2 ms，收益递减）
```

**最佳实践**：先用 `torch.compile` 优化算子，再加 CUDA Graph 消除剩余 overhead：

```python
@torch.compile
def model_forward(x):
    return model(x)

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    output = model_forward(x)

# 推理时
input_tensor.copy_(new_input)
graph.replay()
```

### 3.5 CUDA Graph 实测对比（RTX 3080，1000 个小 kernel）

| 方式 | Launcher 总开销 | 总耗时 | 加速比 |
| --- | --- | --- | --- |
| 每次 sync | 3.09 ms | 11.98 ms | baseline |
| 每 100 次 sync | 2.83 ms | 7.49 ms | 1.60x |
| **CUDA Graph** | **0.73 ms** | **6.31 ms** | **1.90x** |

CUDA Graph 消除了 **76% 的 launcher 开销**。

> **注意**：π₀ 中 kernel 平均执行时间约 1300 μs（大 kernel），launcher 占比仅 0.4%，所以 CUDA Graph 对 π₀ 的实际收益远小于小 kernel 密集场景。π₀ 真正的瓶颈是 GPU 计算本身，而非 CPU 调度。
> 

---

## 四、优化二：计算图变换（减少冗余计算）

这一步通过等价变换消除冗余操作，类似编译器的"常量折叠"，节省约 **7–8 ms**。

### 4.1 折叠 RMS Norm 仿射参数

RMS Norm 后紧接线性层，两者均为线性操作，可以利用结合律合并：

```
原始计算：
  x_norm = x / rms(x) * gamma     ← RMS Norm（含仿射参数 gamma）
  y = x_norm @ W + b               ← Linear

等价变换后：
  W_new = diag(gamma) @ W          ← 预先将 gamma 吸收进 W（只做一次）
  y = (x / rms(x)) @ W_new + b    ← 推理时省去一次 element-wise 乘法 kernel
```

**收益**：每层减少 1 次 element-wise 乘法 kernel，π₀ 共有 27+18 层，合计减少 45 次 kernel 调用。

### 4.2 折叠动作时间编码器

AE 中的动作时间编码器原始结构：

```
原始：
  action(32) → Linear(32→1024) → embed_a(1024) ─┐
                                                  concat(2048) → Linear(2048→1024) → feature
  time(1)   → Linear(1→1024)  → embed_t(1024) ─┘

变换后（推理时）：
  action(32) → Linear(32→1024, 直接合并两次线性) → feature + bias[time_step]
  ↑ 因为 action 分支没有非线性，两次线性可直接折叠为一次
  ↑ time 分支只有 10 种取值，预先打表，折叠为偏置向量 bias[0..9]
```

```python
# 预计算（初始化时）
time_bias_table = {}
for t in range(10):
    time_embed = linear_time(torch.tensor([t]))         # (1024,)
    result = linear_combine(time_embed)                  # (1024,)
    time_bias_table[t] = result  # 存储为常量

# 推理时（只需一次矩阵乘法 + 查表加偏置）
action_feature = linear_action_fused(action)            # 合并后的单次线性
output = action_feature + time_bias_table[current_step] # 查表，零计算
```

**收益**：动作分支 kernel 数从 2 减为 1；时间分支完全消除运行时计算，变为常量偏置加法。

### 4.3 QKV 投影融合

```
原始：
  Q = x @ W_Q    ← kernel 1
  K = x @ W_K    ← kernel 2
  V = x @ W_V    ← kernel 3

融合后：
  QKV = x @ [W_Q | W_K | W_V]   ← 单次大矩阵乘法
  Q, K, V = QKV.split(...)       ← 内存切片，无额外计算
```

**收益**：

- kernel 数从 3 减为 1，减少两次 GPU 调度同步
- 单次大 GEMM 的硬件利用率高于三次小 GEMM（更好的 SM 占用率）
- 可进一步将 RoPE 位置编码融合进矩阵乘法，并预计算 RoPE 权重

---

## 五、优化三：Triton Kernel 深度优化

在消除"低垂的果实"之后，针对每个 GEMM 进行精细化调优，合计再节省约 **8.4 ms**。

### 5.1 调优 GEMM Tile 参数

默认 cuBLAS 按矩阵维度分派预编译 kernel，部分尺寸配置并不最优。手动用 Triton 调优分块策略：

```python
# 以 AE 中高频执行的 GEMM（180次，64×1024×2560）为例
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    BLOCK_M: tl.constexpr = 64,   # 调优后的最优 tile 尺寸
    BLOCK_N: tl.constexpr = 32,
    BLOCK_K: tl.constexpr = 64,
):
    # Tile-based 矩阵乘法，充分利用 SRAM，减少 HBM 访问
    pid = tl.program_id(0)
    ...
```

调优结果（部分关键 kernel）：

| Shape | 执行次数 | cuBLAS 耗时 | Triton 耗时 | Tile 策略 |
| --- | --- | --- | --- | --- |
| 512×2048×32768 | 17 | 7.359 ms | 7.317 ms | fused gate |
| 64×1024×2560 | 180 | 1.718 ms | 1.479 ms | 64,32,64 |
| 512×1152×4304 | 27 | 1.221 ms | 1.074 ms | 64,64,64 |
| 512×4304×1152 | 27 | 1.190 ms | 1.143 ms | 64,64,64,4 |

整体节省约 **1.5 ms**。

> **注意**：LLM 的 transformer 只运行 17 次而非 18 次。这是因为只有 KV cache 会传给 AE，最后一层的 feature 不需要计算，额外节省约 0.7 ms。
> 

### 5.2 Gated FFN 融合（最重要的单项优化）

FFN 使用门控升维结构，原始需要 4 次 kernel 调用：

```python
# 原始（pi0_infer.py 改造前）：4 个 kernel，4 次 HBM 读写
gate = F.linear(x, w_gate)        # kernel 1：读 x, w_gate → 写 gate（HBM）
up   = F.linear(x, w_up)          # kernel 2：读 x, w_up  → 写 up（HBM）
gate = F.gelu(gate)                # kernel 3：读 gate     → 写 gate（HBM）
out  = gate * up                   # kernel 4：读 gate, up → 写 out（HBM）
```

融合后，一次 kernel 完成所有计算，中间结果留在寄存器/SRAM 中：

```python
# 融合后（pi0_infer.py:352）：1 个 Triton kernel，1 次 HBM 写
@triton.jit
def matmul_small_gate(inp, w1, w2, out,
                      M, N, K,
                      BLOCK_M: tl.constexpr,
                      BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr):
    # 加载同一块输入 tile
    a = tl.load(inp_ptr + ...)                 # 读一次 inp
    # 并行计算两个投影（共用同一输入 tile）
    acc1 = tl.dot(a, tl.load(w1_ptr + ...))   # gate projection
    acc2 = tl.dot(a, tl.load(w2_ptr + ...))   # up projection
    # GELU（SiLU 近似）在寄存器中完成
    acc1 = acc1 * tl.sigmoid(1.596 * acc1 * (1 + 0.045 * acc1 * acc1))
    # 合并结果并写回（只写一次）
    output = (acc1 * acc2).to(tl.bfloat16)
    tl.store(out_ptr + ..., output)
```

**为什么有效**：

- kernel 调用 4→1，节省 3 次 GPU 同步开销
- HBM 访问 4→1：中间张量（gate、up）不再落盘到显存，直接在寄存器/SRAM 中流转
- 两个矩阵乘法共用同一输入 tile，输入只需加载一次
- 写回时只写合并后的结果，内存带宽节省约 75%

**实测占比**：`matmul_small_gate` 在 RTX 3080 上占 GPU 时间的 **28.6%**，是第二大瓶颈。

### 5.3 RMS Norm + Matmul + Residual 融合

```python
# 原始：4 步，4 次 HBM 读写
norm = rms_norm(x)           # kernel 1
out  = matmul(norm, w)       # kernel 2
out  = out + bias            # kernel 3
out  = out + residual        # kernel 4

# 融合后（pi0_infer.py:1066）：1 个 kernel
@triton.jit
def matmul_small_bias_res(inp, weight, out, bias, residual,
                          rms_weight, features, ...):
    # RMS Norm 在寄存器中完成
    row = tl.load(inp + ...)
    rms = tl.sqrt(tl.sum(row * row, axis=0) / features + 1e-6)
    inp_norm = row / rms

    # 矩阵乘法
    acc = tl.dot(inp_norm, tl.load(weight + ...))

    # 偏置 + 残差直接加在寄存器中，一次写回
    output = acc + tl.load(bias + ...) + tl.load(residual + ...)
    tl.store(out + ..., output.to(tl.bfloat16))
```

**实测占比**：`matmul_small_res` 在 RTX 3080 上占 GPU 时间的 **34.5%**，是最大瓶颈。

### 5.4 部分 Split-k（处理 SM 分配不均问题）

特殊 GEMM 尺寸 512×1152×1152 存在 SM 分配不均问题：

```
问题：
  使用 64×64 tile → 生成 144 个 block
  RTX 4090 有 128 个 SM
  144 不是 128 的整数倍 → 部分 SM 要多跑一个 block，导致等待

解决方案：拆分为两个可均匀分配的子 GEMM
  子 GEMM 1：512×1152×1024（64×64 tile → 128 blocks，完美分配）
  子 GEMM 2：512×1152×128 （32×32 tile + split-2 → 128 blocks，完美分配）
  两者写入同一 kernel，并行执行（无数据依赖）
```

虽然此项收益不足 0.1 ms，但体现了针对具体硬件拓扑的精细调优思路。

### 5.5 标量操作融合

所有 GEMM 后的标量操作（bias、residual、activation）均与前一 GEMM 融合：

```
bias + residual：合并进矩阵乘法 kernel，一次写回
RMS Norm：先计算 token 级统计量存入独立 buffer，
          在下一 GEMM 的累加阶段除以归一化因子，避免额外读写
```

此步骤减少总体内存占用，综合贡献约 **4 ms**。

---

## 六、性能下界分析（Roofline 模型）

### 6.1 Roofline 公式

对于维度为 N×K×M 的 BF16 GEMM：

$t_{\text{roofline}} = \max\left(\frac{2KM}{T_{\text{bandwidth}}},\ \frac{NKM}{T_{\text{compute}}}\right)$

- 第一项：内存带宽限制（只考虑权重矩阵，激活可驻留 L2）
- 第二项：算力限制

RTX 4090 参数：带宽 1.01 TB/s，实际算力 91.4 TMAC/s（超频至 2.79 GHz）。

### 6.2 各模块理论下界（2 视图）

| 模块 | 理论下界 | 实际耗时 | 效率 |
| --- | --- | --- | --- |
| Vision Encoder | 2.485 ms | 4.059 ms | 61% |
| LLM | 10.727 ms | 12.503 ms | 86% |
| Action Expert | 6.486 ms | 11.001 ms | 59% |
| **合计** | **19.698 ms** | **27.299 ms** | **72%** |

**规律**：

- LLM 大多数操作是**计算密集型**（矩阵大，算力是瓶颈）→ 接近 Roofline
- Vision Encoder 和 AE 大多数操作是**带宽密集型**（小矩阵，HBM 读写是瓶颈）→ 优化空间更大

### 6.3 同步开销（1378 次 kernel 同步）

π₀ 计算图共有 1378 个矩阵乘法操作：

| 同步方式 | 同步时间 | 额外开销 |
| --- | --- | --- |
| PyTorch 顺序 launch | 13.81 ms | +12.92 ms |
| CUDA Graph | 2.61 ms | +1.72 ms |
| 软件屏障（software barrier） | 1.75 ms | +0.86 ms |
| 完全融合（无同步基线） | 0.89 ms | +0 ms |

软件屏障原理：启动数量等于 SM 数的线程块，用全局内存实现跨 block 同步：

```python
# Triton 软件屏障示例
lock_goal += psize
tl.atomic_add(lock_ptr, 1)          # 当前 block 完成
while tl.atomic_or(lock_ptr, 0) < lock_goal:
    pass                             # 等待所有 block 完成
```

实际使用时，软件屏障带来负收益（寄存器压力增加、代码体积变大），仅作为下界估算依据。

加入同步开销后的最终下界（2 视图）：**20.6 ms**，实际 27.3 ms，距极限约 **30%**。

---

## 七、实测性能瓶颈分析（Nsys，RTX 3080）

### 7.1 Nsys 分析方法

```bash
# 运行 profiling
nsys profile --trace=cuda,nvtx,osrt --output=my_profile --stats=true \\
    python3 benchmark.py --model_version pi0 --num_views 2 --chunk_size 63

# 查看 GPU kernel 时间（真正的计算耗时）
nsys stats my_profile.nsys-rep --report gpukernsum | head -n 20

# 查看 CPU API 调用开销
nsys stats my_profile.nsys-rep --report cuda_api_sum | grep Launch
```

> **关键区分**：
> 
> - `CUDA API 行（CPU 端）`：主机调用 CUDA API 的时间，`cudaDeviceSynchronize` 通常占 95%+（正常！CPU 在等 GPU 完成）
> - `CUDA HW 行（GPU 端）`：kernel 在 GPU 上的实际计算时间，这才是真正瓶颈

### 7.2 RTX 3080 实测数据（74.5 ms 推理分解）

```
GPU 计算（CUDA HW）：70 ms  94%  ← 真正瓶颈
  matmul_small_res:       25 ms  34.5%  ← 最大单项（RMSNorm+Matmul+Residual）
  matmul_small_gate:      20 ms  28.6%  ← 第二项（Gated FFN）
  其他 kernels:           25 ms  31.9%

CUDA API 开销（CPU 端）： 3 ms   4%
  cudaGraphLaunch:        0.6 ms  0.8%
  cudaLaunchKernel:       0.9 ms  1.1%
  其他:                   1.5 ms  2.1%

内存传输：< 1 ms  < 1%  （可忽略）
```

**结论**：GPU 计算占 94%，继续优化 CPU/Graph 开销性价比极低。下一步应攻击 matmul kernel 本身，最有效手段是**量化**。

---

## 八、进一步优化方向

### 8.1 量化（优先级最高）

GPU Tensor Core 各代支持的精度：

| 架构 | 代表显卡 | 支持精度 |
| --- | --- | --- |
| Ampere | RTX 3080 | FP16、BF16、INT8、TF32 |
| Ada | RTX 4090 | FP16、BF16、INT8、TF32、**FP8** |
| Hopper | H100 | FP16、BF16、INT8、TF32、**FP8** |

**INT8 量化（RTX 3080 首选）**：

```python
# 使用 bitsandbytes 或 TensorRT INT8
import bitsandbytes as bnb

# 将线性层替换为 INT8
model = bnb.nn.Linear8bitLt(in_features, out_features, has_fp16_weights=False)
```

- 矩阵计算速度是 FP16 的 2–4 倍（INT8 Tensor Core 原生支持）
- 内存减半（FP16 → INT8 节省 50%），带宽压力同步降低
- 预期收益：当前 matmul 占 GPU 时间 63%，INT8 速度约 2 倍 → **节省约 20–25 ms，74 ms → ~50 ms**

**FP8 量化（RTX 4090 / H100 可用）**：

```python
# RTX 4090 / H100 支持 FP8 原生 Tensor Core
# 优点：有指数位，精度损失比 INT8 小，无需精细校准
# 注意：RTX 3080 无 FP8 Tensor Core，用 FP8 反而更慢（需转换为 FP16 再计算）
```

### 8.2 INT4/GPTQ 量化

对于带宽密集型的 AE 部分（权重加载是瓶颈），更激进的 INT4 量化能进一步减少 HBM 读取量：

```python
# 使用 AutoGPTQ
from auto_gptq import AutoGPTQForCausalLM
model = AutoGPTQForCausalLM.from_quantized(model_path, use_triton=True)
```

### 8.3 优化方向优先级

| 优化 | 预期收益 | 适用硬件 | 备注 |
| --- | --- | --- | --- |
| **INT8 量化** | 30–50% | RTX 3080+ | 最优先，直接攻击主瓶颈 |
| **流式推理（Full Streaming）** | 解锁 480 Hz | 所有 | VLM 与 AE 并行执行 |
| **FP8 量化** | 30–50% | RTX 4090 / H100 | 精度损失更小 |
| **知识蒸馏** | 15–30% | 所有 | 减少模型参数量 |
| **FlashAttention-3** | 5–10% | RTX 4090+ | 当前 attention 仅占 3.3% |
| **Tile 参数调优** | 3–7% | 所有 | 适配自己的硬件 |
| **继续优化 CUDA Graph** | 1–3% | 所有 | 已接近极限，性价比低 |

---

## 九、全流式推理框架（Full Streaming Inference）

### 9.1 并发执行收益

AE（IO 密集）与 VLM（计算密集）在硬件层面互补，可并发执行：

```
顺序执行 VLM + 10 次 AE：27.3 ms
并发执行 VLM + 10 次 AE：26.3 ms（-1 ms）
并发执行 VLM + 16 次 AE：32.7 ms（触及 33 ms 边界）

→ 每秒可支撑 30 次 VLM + 480 次 AE
```

### 9.2 三层反馈回路

```
480 Hz ← Action Expert（IO 密集）  处理力传感器，<2 ms 延迟
 30 Hz ← Vision Encoder + LLM      处理图像，<33 ms 延迟
  1 Hz ← LLM 文本生成（搭载帧编码）  任务规划，30 token/s
```

**高频控制信号来源**：力传感器（>2 KHz）、电机电流（>1 KHz）、电阻式触觉传感器。

**关键设计**：AE 通过渐进式流匹配（类自回归解码）生成动作——每一步 AE 只生成部分动作序列，而非等待全部 10 步去噪完成后再输出。新传感器数据通过独立 CUDA stream 的 memcpy 注入，不打断主推理流程。

---

## 十、经验总结

### 10.1 收益递减规律

```
Naive PyTorch:   106.5 ms
+ Triton:         ~80 ms   ← 第一步，收益最大（直接换更好的 kernel）
+ torch.compile:  ~45 ms   ← 第二步
+ CUDA Graph:     43.5 ms  ← 第三步，收益递减（已有 compile 时仅 3%）
+ 计算图简化:     35.7 ms
+ Kernel 调优:    27.3 ms

后续每一步都建立在前一步基础上，越往后优化越难
```