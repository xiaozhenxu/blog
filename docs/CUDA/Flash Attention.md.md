# Flash Attention

# 1. 背景

在 Transformer 结构当中，标准的 attention 计算公式如下

$$
 \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V 
$$

从公式的实现上来看，可以划分为三个步骤，即 矩阵乘 - $softmax$ - 矩阵乘。如果用 PyTorch 来实现，如下。

```python
from torch import Tensor

def attention(query: Tensor, key: Tensor, value: Tensor):
	# query [B, Lq, D]
	# key   [B, Lk, D]
	# value [B, Lk, D]
	d = query.shape[-1]
	scale = d ** (-0.5)
	score = (query @ key.transpose(-1, -2)) * scale
	score = score.softmax(dim=-1)
	out = score @ value
	return out
```

在大模型推理长序列场景下，传统实现会显式构造/存储 $*QK^T*$ 和 $softmax$ 权重矩阵，他们的显存占用将会达到 $*O(n^2))$，*并且对于这部分显存读取将导致带宽瓶颈。

# 2. Flash Attention 算法

## 2.1 算法实现原理

本文不涉及 safe softmax 和 online softmax 等数学公式推导，而专注于算法原理和算法实现。

Flash Attention 的算法原理可以概括为：

- 分块计算（tiling）：把 $*K,V$* 按序列维度切成小块，与 $Q$ 的块配对，在块内完成点积与后续计算。
- online softmax：对 query 每行维护3个累积量：当前最大值 $m$、归一化分母 $\ell = \sum e^{s-m}$ 、加权和值 $a = \sum e^{s-m} V$。每处理一个新块就更新 $m$  $\ell$  $a$，最后输出为 $O = a/\ell$ 。因此不需要保存整行/整块的注意力权重。
- 算子融合与低访存：在一个 fused kernel 中把 三个计算步骤 串起来做，尽量只在片上存储中间量，减少对显存的读写，从而显著提速并降低显存占用。

![image.png](images/Flash%20Attention/image.png)

从图中可以看出，Flash Attention 的计算是分步迭代的，这里对 *Q*,*K*,*V*  矩阵的 shape 定义如下：

| quey | [SeqLength_Q,  DIM] |
| --- | --- |
| key | [SeqLength_KV, DIM] |
| value | [SeqLength_KV, DIM] |

每个 BLOCK 负责输出矩阵 value 的一部分，因此每个 BLOCK 对应处理的矩阵 shape 定义如下：

| block query | [BLOCK_Q,  DIM] |
| --- | --- |
| block key | [SeqLength_KV, DIM] |
| block value | [SeqLength_KV, DIM] |

## 2.2 Python 伪代码实现

下面的 Python 伪代码实现了 Flash Attention 2 论文中描述的算法。每个线程块（block）负责处理 Q 的一个分块（这里指的是在 seqlen 维度上分块），并且会遍历整个 K V。

**Notice: 这样的计算方法使得每个 block 负责计算的部分是独立的，不需要通讯。**

```python
scale = DIM ** -0.5
for b_idx in range(B):
	for tile_Q_idx in range(Lq // BLOCK_Q):
		### 下面是每个核函数执行的内容
		tile_O = torch.zeros(BLOCK_Q, DIM)
		tile_Q = load_Q(b_idx, tile_Q_idx) # (BLOCK_Q, DIM)
		
		for tile_KV_idx in range(Lk // BLOCK_KV):
			### S = Q @ K.T
			### (BLOCK_Q, DIM) x (BLOCK_KV, DIM).T -> (BLOCK_Q, BLOCK_KV)
			### kv 矩阵也是在 seqlen 维度分块，因此这里需要使用 online softmax
			tile_Q
			tile_K = load_K(b_idx, tile_KV_idx)
			tile_S = tile_Q @ tile_K.T
			tile_S = tile_S * scale
			
			# online softmax and rescale tile_O
			...
			
			### O = P @ V
			tile_P                               # (BLOCK_Q, BLOCK_KV)
      tile_V = load_V(b_idx, tile_KV_idx)  # (BLOCK_KV, DIM)
      tile_O += tile_P @ tile_V            # (BLOCK_Q, DIM)
```

上述代码的实现包含了一个前提，也就是 DIM 较小，这样才能在整个 kernel 函数执行期间将 tile_Q 一直保存在寄存器内存当中。现在主流模型的标准配置就是 $head\_dim = 128$，因为它可以完美契合 NVIDIA GPU 中 Tensor Core 的计算粒度。

Online-Softmax 简单实现可以如下，对应的是一个 ThreadBlock（线程块）的工作流程：固定负责一部分 Q（`tile_Q`），然后遍历所有的 K 和 V（`tile_K` 和 `tile_V`)。

```python
att_scale = DIM ** -0.5

# attention state
m = torch.zeros(BLOCK_Q)              # 运行中的最大值 (Running Max)
tile_O = torch.zeros(BLOCK_Q, DIM)    # 运行中的分子 (Numerator)
sumexp = torch.zeros(BLOCK_Q)         # 运行中的分母 (Denominator)

for _ in range(Lk // BLOCK_KV):
  # 1st MMA
  tile_S = tile_Q @ tile_K.T  # [BLOCK_Q, BLOCK_KV]
  tile_S = tile_S * att_scale

  # online softmax
  tile_max = tile_S.amax(dim=-1)  # [BLOCK_Q]
  new_m = torch.maximum(m, tile_max)
  tile_P = torch.exp(tile_S - new_m.unsqueeze(-1))

  # rescale
  scale = torch.exp(m - new_m)
  tile_O *= scale.unsqueeze(-1)
  sumexp = sumexp * scale + tile_P.sum(dim=-1)
  m = new_m  # save new max

  # 2nd MMA
  tile_O += tile_P @ tile_V  # [BLOCK_Q, DIM]

# apply normalization
tile_O /= sumexp.unsqueeze(-1)	

```

初始化得到的三个变量分别代表下面的意思

- `m` 保存目前为止遇到的最大的 Attention Score。用于防止指数溢出。
- `tile_Q` 保存目前为止累加的加权 Value 和（即 softmax 公式中的分子部分 $\sum e^{S} V$）
- `sumexp` 保存目前为止的指数和（即 Softmax 公式的分母部分 $\sum e^{S}$）

为了方便后续的数学表达，首先定义下面的数学符号

- $Q$ 当前线程块负责的 Query 块
- $K_j,V_j$ 第 j 个 Key/Value 分块
- $S^j$ 第 j 个分块的 Attention 值，shape 为 `(BLOCK_Q, BLOCK_KV)`
- $P^j$  第 j 个分块 Attention 值的指数结果，shape 为`(BLOCK_Q, BLOCK_KV)`
- $m^j$ 处理完第 j 个分块后当前的 Attention 值局部最大值，shape 为 `(BLOCK_Q)`
- $O^j$ 处理完第 j 个块后的输出累加值（对应代码中的 `tile_O`）
- $l^j$ 处理完第 j 个块后指数和（对应代码中的 `sumexp` ）shape 为 `(BLOCK_Q)`

对于第 j 个分块$(j = 1, 2,…)$，执行以下的步骤

1. 计算 Attention 分数，对应代码中的 `tile_S = tile_Q @ tile_K.T` 此处省略了缩放系数的处理

$$
S^j=QK^T_j
$$

1. 计算局部最大值

$$
\tilde{m}^j = rowmax(S^j)
$$

1. 更新全局最大值

$$
m^j=max(m^{j-1},\tilde{m}^j)
$$

1. 计算当前块的 P，这里是基于新的局部最大值 $m^j$ 计算得到的，对应代码中的`tile_P = torch.exp(tile_S - new_m.unsqueeze(-1))`

$$
P^j=exp(S^j-m^j)
$$

1. 计算修正系数，对应的代码是 `scale = torch.exp(m - new_m)`

$$
\alpha^j=exp(m^{j-1}-m^j)
$$

1. 更新分母，对应的代码是 `sumexp = sumexp * scale + tile_P.sum(dim=-1)` 
    
    新的分母 = (旧分母 x 修正系数) + 当前块的指数和
    
    $$
    l^j=l^{j-1}*\alpha^j+rowsum(P^j)
    $$
    
2. 更新分子，新的分子 = (旧分子 x 修正系数) + (当前块的P x 当前块的V)

$$
O^j=O^{j-1}*\alpha^j+P^jV_j
$$

1. 在所有轮数迭代完成之后，用最终的分子分母得到结果

# 3. Version 1 - 基础实现

通常 MMA 的实现遵循下面的步骤

1. 使用 `cp.async` 从 global memory 加载到 shared memory
2. 使用 `ldmatrix` 从 shared memory 加载到寄存器文件
3. 调用 `mma.m16n8k16` 实现 BF16 的矩阵乘加操作

## 3.1 Global to Shared memory data transfer

```cpp
#include <cuda_bf16.h>

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline
void global_to_shared(uint32_t dst, const nv_bfloat16 *src, int src_stride, int tid) {
  // cp.async.cg.shared.global 指令为每个 cuda 线程执行 16 字节的传输，即 8 个 bf16
  constexpr int num_elems = 16 / sizeof(nv_bfloat16);
  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const uint32_t dst_addr = dst + (row * WIDTH + col) * sizeof(nv_bfloat16);
    const nv_bfloat16 *src_addr = src + (row * src_stride + col);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
  }
}
```

上面的代码是使用内联汇编 `cp.async.cg.shared.global` ****来实现的，这条 PTX 指令为每个 CUDA 线程执行 16 字节的数据传输，即 8 个 BF16 元素（`num_elems = 16 / sizeof(nv_bfloat16)`）。为了确保合并内存访问（coalesced memory access），连续的线程将负责处理连续的 8xBF16 数据组。

循环 `for (int iter = 0; iter < num_iters; iter++)` 这样写是为了让编译器 (nvcc) 能够完全展开（unroll）该循环。`num_iters` 在编译时是已知的（由 `constexpr` 保证）。如果我们在循环中混入 `tid`（对编译器来说这是一个“动态”变量），循环就无法被展开，即使我们知道关于该变量的某些约束条件（例如 `tid < TB_SIZE`）。

共享内存指针 `dst` 的数据类型是 `uint32_t`。这是有意为之的。几乎所有的 PTX 指令都期望共享内存地址处于共享状态空间（shared state space）中。我们可以使用 `static_cast<uint32_t>(__cvta_generic_to_shared(ptr))` 将 C++ 指针（通用地址）转换为共享状态空间地址。这一步是在 `global_to_shared()` 外部完成的。

为了完成 `cp.async` 的使用，我们还需要添加以下内容：

- `cp.async.commit_group` **(PTX)**：将之前发出的所有 `cp.async` 指令提交到一个 `cp.async` 组中。这个组将作为同步的单位。
- `cp.async.wait_all` **(PTX)**：等待所有已提交的组完成。
- `__syncthreads()`：确保（一个线程块中的）所有线程在读取共享内存中加载的数据之前都到达这里（因为一个线程可能会读取另一个线程加载的数据）。更重要的是，这会将新数据的可见性（visibility）广播给线程块中的所有线程。如果没有 `__syncthreads()`，编译器可以自由地将内存访问优化掉。

```cpp
// nv_bfloat16 *Q
// uint32_t Q_smem
// constexpr int TB_SIZE = 32 * 4
// constexpr int DIM = 128

global_to_shared<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, DIM, tid);
asm volatile("cp.async.commit_group");
asm volatile("cp.async.wait_all");
__syncthreads();
```

## **3.2 Shared memory to Register memory data transfer**

在实现全局内存向共享内存的数据迁移的时候，我们从一个 thread block 和每个 cuda thread 的角度来考虑分块。但是，在做共享内存向寄存器文件的数据迁移的时候，因为寄存器数据后续服务于 MMA 指令，这使得我们以 warp 为单位来考虑数据切块。

![image.png](images/Flash%20Attention/image%201.png)

解释一下上面这张图，首先解释一下各个变量所指代的意义

- `BLOCK_Q` kernel 函数当中每个 block 负责 query 的 seq_len
- `BLOCK_KV` 与 `BLOCK_Q` 不同，是在每个 kernel 函数内部对 kv 矩阵 seq 维度上的切分

让线程块中的每个 warp 处理 tile_Q 的一部分，即沿着 Q 的序列长度维度进行切分，因此不同的 warp 会访问 tile_Q 的不同部分。但是，在 KV 序列长度的循环中，每个 warp 都会访问到每一个 tile_K tile_V 分块。

由于我们将使用`mma.m16n8k16` 来计算矩阵乘法，每个 16x8 的 output tile 需要 16x16 的 A tile 和 8x16 的 B tile。 `ldmatrix` 可以加载一个、两个或者四个由 16bits 元素组成的 8x8 tile

- A_tile (16x16) 需要四个 8x8 分块 - > 使用 `ldmatrix.x4`
- B_tile (8x18) 需要两个 8x8 分块 - > 使用 `ldmatrix.x2`

![image.png](images/Flash%20Attention/image%202.png)

根据上图，想要使用 `ldmatrix` ，每个线程需要提供某一行的地址。线程 0-7 选择第 1 个 8x8 分块，线程 8-15 选择第 2 个 8x8 分块，依次类推。

下面是使用 `ldmatrix_x4` 实现 Q 从 shared_memory 向 register 搬运的代码

```cpp
 // 定义 Tensor Core 的计算粒度
 constexpr int MMA_M = 16;
 constexpr int MMA_N = 8;
 constexpr int MMA_K = 16;
 
 // Q_smem 是共享内存的起始地址指针
 uint32_t Q_smem;
 
 // Q_rmem 是寄存器数组
 // [WARP_Q / MMA_M] 垂直方向需要切几刀（例如warp负责64行，每刀16行，得到4块）
 // [DIM / MMA_K] 水平方向需要切几刀（例如 dim=128, 每刀16列，得到8块）
 // [4] 每个 tile 为 16x16，half 类型(2字节），每个线程需要保存8个元素，用到4个32位寄存器
 uint32_t Q_rmem[WARP_Q / MMA_M][DIM / MMA_K][4];
 // 双层循环：将大矩阵切分成 16*16 的 tile
 for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
  for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
	  const int row = (warp_id * WARP_Q) // 1. 跳过之前 warp 负责的行
								  + (mma_id_q * MMA_M) // 2. 跳过当前 warp 已经处理过的 16x16 块
								  + (lane_id % 16); // 3. warp 内的行偏移
		const int col = (mma_id_d * MMA_K) // 1. 跳过已经处理的列块
		              + (lane_id / 16 * 8); // 2. warp 内的列偏移
		// 计算最终的共享内存地址 (转成字节偏移)
		const uint32_t addr = Q_smem + (row * DIM + col) * sizeof(nv_bfloat16);

		// 执行 PTX 指令
		ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], addr);
 }
```

![image.png](images/Flash%20Attention/image%203.png)

注意，前面说到，“每个线程需要提供某一行的地址”。这里做出具体说明，一个 16x16 的矩阵块，划分为 4 个 8x8 的更小矩阵块，一个 warp 当中每 8 个线程处理其中的某个小矩阵块，即 1x8 的块，这也就是针对某个线程需要具体计算得出的某一行的地址。

![image.png](images/Flash%20Attention/image%204.png)

### Online softmax - CUDA C++

**Row max**

首先看对 `tile_S` 取行最大值，这里有个想当然的做法，将一个 `tile_S` 计算出来之后，放回 shared memory，然后再对这块 shared memory 处理得到 row max。

但是，从文档的做法来看，他是打算直接在寄存器当中去做处理，一个 `tile_S` 是通过**多个 warp 并行以及 warp 内的多次串行 MMA 指令**操作得到的。因此，针对 `tile_S` ，每个线程在寄存器中保存了下面的数据

```python
float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4];
```

![image.png](images/Flash%20Attention/image%205.png)

前面的两个维度表示 1 个 warp 处理的数据被 mma 切块；

 **`4`** 对应上图中的 c0、c1、c2、c3，也就是说，每个线程持有来自 2 行的 2 个连续元素。为了在（MMA 输出分块的）行内进行归约（reduction），我们首先对单个线程持有的 2 个连续元素进行归约，然后在 4 个线程组成的组内（即 T0-T3，T4-T7 等）进行归约。然而，行归约实际上是针对整个 **`tile_S`** 进行的，因此我们还需要遍历 **`S_rmem`** 中的 **`BLOCK_KV / MMA_N`** 这一维。这一步可以在进行 4 线程组归约之前，与线程级归约结合起来完成。

![image.png](images/Flash%20Attention/image%206.png)

这个地方引入 `__shfl_xor_sync` ，可以直接获取指定线程寄存器当中的值。在一个 MMA tile 中，4 个连续的线程编号负责一行数据，比如编号 id 为 0,1,2,3 的线程。分成两个步骤，第一步先比较 id 为 0 & 1 和 2 & 3 线程，第二步比较第一步获得的最大值，最终 4 个线程都取得最大值。

**Rescaling**

在计算完当前分块(tile)的行最大值后，可以计算得到输出结果的重缩放因子，以及归一化项（即每一行的指数和）。

```cpp
// 更新行最大值
// this_rowmax 是当前分块计算出的局部最大值
// rowmax 是之前所有分块累计的全局最大值
this_rowmax[0] = max(this_rowmax[0], rowmax[mma_id_q][0]);
this_rowmax[1] = max(this_rowmax[1], rowmax[mma_id_q][1]);

// 计算针对之前输出 O 的重缩放因子
float rescale[2];
// 因子 = exp(旧最大值 - 新最大值)
rescale[0] = __expf(rowmax[mma_id_q][0] - this_rowmax[0]);
rescale[1] = __expf(rowmax[mma_id_q][1] - this_rowmax[1]);

// 遍历输出矩阵 O 的所有部分进行重缩放
for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
  // 0和1属于第一行，乘第一行的因子
  O_rmem[mma_id_q][mma_id_d][0] *= rescale[0];
  O_rmem[mma_id_q][mma_id_d][1] *= rescale[0];
  // 2和3属于第二行，乘第二行的因子
  O_rmem[mma_id_q][mma_id_d][2] *= rescale[1];
  O_rmem[mma_id_q][mma_id_d][3] *= rescale[1];
}

// 保存新的行最大值供下一次迭代使用
rowmax[mma_id_q][0] = this_rowmax[0];
rowmax[mma_id_q][1] = this_rowmax[1];
```

## Version 2 - XOR Swizzling

### Profiling

![image.png](images/Flash%20Attention/image%207.png)

上图的横轴表示平均每个调度器在任意一个时钟周期内，有多少个 Warp 处于该状态。

- **Stall Math Pipe Throttle 等待硬件计算单元**
- **Stall Short Scoreboard 等待从共享内存或者 L1 Cache 上读取数据**
- **Stall Long Scoreboard  等待从全局内存中读取数据**

### Swizzle

通过改变数据在 Shared Memory 中的物理存储地址（而不改变逻辑上的行列关系），来避免 Bank Conflict（如果一个 Warp 中的多个线程同时访问同一个 Bank 的不同地址，访问就会串行化，导致性能下降）。

而使用 `ldmatrix` 指令从 Shared Memory 读取矩阵块时，如果数据是标准的“行主序”排列，会导致严重的 Bank Conflict。

```cpp
template <int STRIDE>
__device__ uint32_t swizzle(uint32_t index) {
	// 如果是一个跨度比较小的矩阵，则不需要 swizzle
	if constexpr(STRIDE == 16) return index;
	
	// 1. 计算当前地址属在逻辑上属于哪一行
	// index 是字节偏移量，STRIDE 是一行的字节数
	uint32_t row_idx = index / STRIDE % 8；
	
	// 2. 计算 xor 偏移量
	
	// 3. 执行 xor 操作
	
	
		
}
```

<aside>
💡

注意点

STRIDE 表示每一行的字节数量。

如果 STRIDE  数值比较小，这意味着数据比较紧凑，不需要 swizzle。

如果 STRIDE 为 16，则每行有 4 个 bank，任意连续的 8 行数据（Row 0～7）完全分布在 Bank 0 到 Bank 31 上，没有任何的重叠。因此，在使用 ldmatrix 指令搬运数据的时候将天然的 conflict free

</aside>

# FQA

1. FlashAttention 主要是解决了什么问题？
    - 将原本的 3 个算子融合为 1 个算子，原本需要 6 次针对 HBM 内存的读取写入操作，而现在只需要 2 次，减少访存耗时，增加了计算强度(即每个读入数据的计算步骤增多了)
    - 不需要将中间结果（即 shape 是`(qSeqlen, kSeqlen)` 的矩阵）存储进 HBM 中，节约了显存