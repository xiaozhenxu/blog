# LDG vs cp.async vs TMA：GPU 数据搬运技术对比详解

## 引言

现在 GPU 的算力（FP16/BF16 Tensor Core TFLOPS）每代都在翻倍增长，但 HBM 带宽的增速远跟不上算力。这意味着越往后，能否把 tensor core 喂饱（即 G→S→Reg 的搬运效率）越将成为决定 kernel 性能的关键。Kernel 中一次 load 指令从 HBM 出发，要依次经过 L2→L1/SMEM→Register 多个层级，每一层的延迟、带宽、并发度都不同；如何编排这条搬运链路，是写好 GEMM、Attention 等核心算子的核心命题之一。

围绕这一问题，NVIDIA 在过去几代架构里给出了三种代际递进的解法：

- LDG（Volta 起，所有架构都通用）：经典同步 load，线程亲自把数据从 global 搬到 register，再写入到 shared memory。
- cp.async（Ampere SM_80 引入）：半异步搬运，数据 G→S 直通，不需要寄存器，搬运和计算可以流水重叠。
- TMA（Hopper SM_90 引入）：全异步硬件 DMA，单线程发 1 条指令即可搬运整个 tile。地址计算、边界处理、swizzle 全部由硬件完成。

**三代演进的核心设计原则：让搬运尽可能少占用线程、寄存器和发射带宽，把更多的硬件资源留给计算。**

## Part 1: LDG（同步 Load）

LDG（Load Global）是最朴素的搬运方式：每个线程发射 LD.E 指令把全局内存数据读进寄存器，再通过 ST.S 写入到 shared memory。整个过程是同步的，线程发出 load 后会 stall 在 LSU 上，直到数据回到寄存器才能继续往下执行。

下面是一个典型的实现：256 个线程协同搬运一块 64x64 的 bf16 tile（共 4096 个元素），每个线程平均搬 16 个，使用经典的 stride-by-block 循环模式（线程 tid 处理元素 tid，tid+THREADS，tid+2*THREADS,…），能够保证 warp 内访存合并。

```cpp
  // 256 个线程搬运矩阵 shape 为 64*64 的数据
  #define BM 64
  #define BK 64
  #define THREADS 256

  __global__ void copy_kernel(
      const bf16* __restrict__ A,   // 输入矩阵，row-major，M*K
      const int M,
      const int K)
  {
      __shared__ bf16 smem[BM * BK];
      const int tid = threadIdx.x;
      const int by = blockIdx.y;          // tile 在 M 方向上的索引
      const int bx = blockIdx.x;          // tile 在 K 方向上的索引
      const bf16* begin_A = A + by * BM * K + bx * BK;

      // stride-by-block 搬运：每线程搬 BM*BK/THREADS = 16 个 bf16
      for (int i = tid; i < BM * BK; i += THREADS) {
          int row = i / BK;
          int col = i % BK;
          smem[i] = begin_A[row * K + col];   // 隐式：LD → Reg → ST
      }

      __syncthreads();   // 等待 block 内 256 个线程全部搬完

      /* GEMM */
  }
```

### LDG 存在的问题

1. 数据通路绕路：load 的中间结果必须落到寄存器再写出，路径是 G→Reg→S。这将占用寄存器文件和读写带宽，一旦 tile 大、每线程搬运量多，活跃寄存器数上升，occupancy 会被压低。
2. 线程被阻塞：LD.E 是同步指令，编译器虽然会通过指令重排尽量隐藏延迟，但本质上线程仍要等数据回到 Register 才能继续；__syncthreads 之前线程做不了别的事。
3. 搬运与计算串行：必须等所有线程的搬运都完成（__syncthreads）才能进入 GEMM 主循环。时间线上是"先搬完，再算"的串行模型，HBM 带宽和 tensor core 在不同时段被使用，无法重叠。

这三点正是后续 cp.async 与 TMA 的改进点：分别对应绕开寄存器、异步发指令、搬运/计算 pipeline 重叠。

## Part 2: cp.async（Ampere 架构的半异步）

Ampere（SM_80）引入了 `cp.async` 指令，本质是一条异步拷贝指令：从 global memory 直接搬到 shared memory，完全绕开寄存器。这对应了 LDG 存在的问题的前两项：

- 数据通路：G→S 直通，不再通过 Register，occupancy 不再被数据搬运所压低。
- 执行模型：线程发出 `cp.async` 后立刻返回，不阻塞，数据由 LSU 后台搬运。线程可以继续做地址运算、发下一条 `cp.async` ，甚至进入下一段计算逻辑。

加上配套的 `commit_group` / `wait_group` 同步原语，软件可以构造多 stage pipeline，让“上一轮计算”与“下一轮搬运”在时间线上重叠，而这是 LDG 完全做不到的。

```cpp
  // 256 个线程搬运矩阵 shape 为 64*64 的 bf16 数据
  #define BM 64
  #define BK 64
  #define THREADS 256

  // cp.async 只能搬运 4 / 8 / 16 B 三种粒度，这里选最大的 16B（8 个 bf16）
  constexpr int ELEMS_PER_LDG = 8;

  __global__ void cp_async_copy_kernel(
      const bf16* __restrict__ A,   // 输入矩阵，row-major，M*K
      const int M,
      const int K)
  {
      __shared__ bf16 smem[BM * BK];
      int tid = threadIdx.x;
      int by = blockIdx.y;
      int bx = blockIdx.x;

      // 通用 64-bit 指针 → 32-bit shared 地址（PTX shared state space 要求）
      uint32_t smem_addr = __cvta_generic_to_shared(smem);
      const bf16* begin_A = A + by * BM * K + bx * BK;

      // 每线程每轮搬 8 个 bf16，4 轮覆盖整个 64x64 tile
      for (int i = tid * ELEMS_PER_LDG; i < BM * BK; i += THREADS * ELEMS_PER_LDG) {
          int row = i / BK;
          int col = i % BK;

          asm volatile(
              "cp.async.cg.shared.global [%0], [%1], 16;\n"
              // ↑ cg = cache global：只缓存在 L2，不进 L1
              :: "r"(smem_addr + i * sizeof(bf16)),
                 "l"(begin_A + row * K + col));
      }

      asm volatile("cp.async.commit_group;\n");   // 已发的 cp.async 打包成一组
      asm volatile("cp.async.wait_group 0;\n");   // 等到剩 0 组未完成
      // wait_group 只保证当前线程的 cp.async 完成，不保证其它线程的，
      // 因此还需要一次 __syncthreads() 让整个 block 看到完整 smem
      __syncthreads();

      /* GEMM */
  }

```

### 关键点

1. 为什么每线程一次搬 16B？

`cp.async` 只支持 4 / 8 / 16 字节三种粒度。16B 是最优选择：

- 一个 warp 32 个线程一次发出 32×16B = 512B 请求，正好对齐 L2 sector 的访问粒度；
- 16B/线程 = 8 个 bf16，配合 stride-by-block 循环可以让 256 线程一次搬完 2048 个元素，4 轮覆盖 64×64 tile；
- 大于 16B 硬件不支持；小于 16B 会让发射量暴增，issue 端反而成瓶颈。

1. 为什么需要 `__cvta_generic_to_shared`？

cp.async 的目的地址必须是 32-bit shared memory 地址（PTX 的 shared state space 是独立编址的），不能直接传 C++ 通用指针。这条内置函数把 64-bit 通用指针转成 32-bit shared 地址。

1. `cp.async.cg` 的 cg 是什么？

cg = cache global，告诉硬件这条 load 只缓存在 L2，不进 L1。GEMM 这类一次性消费的数据走 cg 比走 ca（cache all）更友好：避免污染 L1，把宝贵的 L1/SMEM 容量留给 shared memory。

1. `commit_group` / `wait_group` 的语义
- `cp.async.commit_group`：把当前线程已发但未确认的所有 `cp.async` 指令打包成一个"组"。
- `cp.async.wait_group` N：等待直到只剩最近 N 组未完成（写 0 就
是全部等完）。

注意 `wait_group` 是线程级同步，只保证当前线程自己发的 `cp.async` 完成，不保证同 block 其他线程的。所以后面还要一次 `__syncthreads()` 让整个 block 看到完整的 smem。

### `cp.async` 仍没解决的问题

1. 每个线程都要发指令：256 线程 * 4 轮 = 1024 条 `cp.async` 指令；issue 端被搬运占用，留给 mma 的发射 slot 减少。
2. 地址、边界、swizzle 仍由软件处理：在循环中的 `i / BK` `i % BK` 全是软件计算的；如果要做 swizzle 让 smem 布局适配 mma，还要在循环中手写 xor 逻辑。
3. 同步粒度粗： `wait_group` 只能按照指令组等，不能按照字节等；在多个 producer 场景下（多个 warp 各自搬运一部分）不好做细粒度的协调

这三个问题都在 Hopper TMA 中得到了解决。

## Part 3: TMA（Hopper 架构下的全异步硬件搬运）

### 整体工作流

`cp.async` 的“半异步”还是要依靠每个线程发指令。而 TMA 只需要单个线程发送指令即可：

- **Host 端**：用 `cuTensorMapEncodeTiled` 构造一个 `CUtensorMap`描述符，告诉硬件"这个全局张量的形状是 M×K，一个 tile 是 BM×BK，stride 多大，是否 swizzle……"。这个描述符是常量，构造一次，全程复用。
- **Device 端**：
    1. 1 个线程发 1 条 `cp.async.bulk.tensor` 指令，硬件 DMA 引擎接管所有的地址计算、边界裁剪、swizzle 重排；
    2. 同时硬件按字节扣减一个 `mbarrier`（驻留在 smem 的 8 字节状态字）的事务计数；
    3. block 内所有线程在 mbarrier 上自旋等待，事务计数归零→相位翻转→线程放行→数据已就绪。

整个流程的示意图如下

```cpp
Host: cuTensorMapEncodeTiled()   // 构造 CUtensorMap

Device:
  tid==0:
    mbarrier.init                // (count=1)
    mbarrier.arrive.expect_tx(bytes=tile_bytes)
    cp.async.bulk.tensor         // 硬件后台搬运

  mbarrier.try_wait.parity       // 所有线程等 phase 翻转
```

### Host 端：构造 CUtensorMap

`CUtensorMap` 是一个 128 字节的硬件描述符，包含了张量的形状 / stride / tile shape / swizzle 等元信息。它不是普通指针，需要通过调用函数 `cuTensorMapEncodeTiled` 获得。

```cpp
#define BM 64
#define BK 64

// 通过 driver API 动态获取 cuTensorMapEncodeTiled 函数指针
PFN_cuTensorMapEncodeTiled cuTensorMapEncodeTiled_fn = nullptr;
cudaDriverEntryPointQueryResult drv_status;
cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",
(void**)&cuTensorMapEncodeTiled_fn,
cudaEnableDefault, &drv_status);

CUtensorMap tma_map{};
void* gmem_ptr = (void*)A;

// 全局张量 row-major M*K，TMA 维度顺序按 fastest-varying 在前：
//   dim0 = K (内层连续维)，dim1 = M
uint64_t gmem_shape[2]  = { (uint64_t)K, (uint64_t)M };
// stride 单位是 byte，只需要给除最内维以外的 stride
uint64_t gmem_stride[1] = { (uint64_t)K * sizeof(bf16) };
// tile 形状，与 smem buffer 对应：BK 在内层，BM 在外层
uint32_t box_shape[2]   = { (uint32_t)BK, (uint32_t)BM };
// 采样步长，dense 矩阵设 {1, 1} 即可
uint32_t elem_stride[2] = { 1, 1 };

CUresult res = cuTensorMapEncodeTiled_fn(
	&tma_map,
	CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
	/*tensorRank=*/2,
	gmem_ptr,
	gmem_shape,
	gmem_stride,
	box_shape,
	elem_stride,
	CU_TENSOR_MAP_INTERLEAVE_NONE,
	CU_TENSOR_MAP_SWIZZLE_NONE,        // 真实场景常用 128B swizzle 配合 wgmma
	CU_TENSOR_MAP_L2_PROMOTION_NONE,
	CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
assert(res == CUDA_SUCCESS);
```

注意：`CUtensorMap` 必须按值传给 kernel 并加 **grid_constant** 修饰（或放到 constant memory），不能传指针。

### Device 端：kernel 主体

```cpp
**global** void tma_copy_kernel(const **grid_constant** CUtensorMap tma_map_A)
{
	const int tid = threadIdx.x;
	const int bx  = blockIdx.x;
	const int by  = blockIdx.y;
  
  // smem 必须 16B 对齐；配合 128B swizzle / wgmma，直接按 128B 对齐最稳妥
  __shared__ alignas(128) bf16 smem[BM * BK];
  // mbarrier 是 64-bit 状态字，PTX 要求 8B 对齐
  __shared__ alignas(8) uint64_t bar;

  // 1) 由单线程初始化 mbarrier 并发起 TMA
  if (tid == 0) {
      mbarrier_init(&bar, /*count=*/1);
      mbarrier_arrive_expect_tx(&bar, BM * BK * sizeof(bf16));
      // 坐标以"元素"为单位，维度顺序与 box_shape 一致：(K, M)
      tma_load_2d(&tma_map_A, smem, &bar,
                  bx * BK,    // crd_x：K 维起点
                  by * BM);   // crd_y：M 维起点
  }
  __syncthreads();   // 让所有线程看到已初始化好的 bar

  // 2) 全 block 等待硬件 DMA 完成（事务计数归零）
  mbarrier_wait(&bar, /*phase=*/0);

  /* GEMM */
}
```

整段 kernel 没有任何地址计算、边界判断、swizzle 编码，这些全都被硬件根据 `CUtensorMap` 传入的信息实现。

### mbarrier：异步同步的核心

`mbarrier` 是一个驻留在 shared memory 里的 8 字节状态字，由硬件维护，作为 TMA 与线程之间的同步桥梁。其状态字编码三个字段：

| arrival count  | 还差多少线程到达 |
| --- | --- |
| transaction count | 还差多少字节没到 |
| phase bit | 相位翻转标志 |

完整的流程分四步：初始化 → 登记字节数 → TMA 搬运（硬件自动扣减）→ 线程等 phase 翻转。下面分别介绍每一步对应的 PTX 原语。

**Step 1: 初始化**

`mbarrier.init.shared::cta.b64 [%0], %1;`

| 输入 | 含义 | 类型 |
| --- | --- | --- |
| [%0] = bar_addr | mbarrier 对象的 shared memory 地址（32-bit） | .r(u32) |
| %1 = count | expected arrival count——多少个线程 arrive 后才会触发 phase 翻转 | .r(u32) |

```cpp
// 这个 barrier 期望 1 次 arrival (因为让 tid==0 这一个线程负责 arrive)
mbarrier_init(&bar, 1);  // 可以通过这样的方式来调用

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t count) {
	uint32_t bar_addr = __cvta_generic_to_shared(bar);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
               :: "r"(bar_addr), "r"(count));
}
```

**Step 2: 登记待传输数据的总字节数量**

`mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;`

这一条指令同时做两件事：

- arrive 一次：arrival count 减 1，表示当前线程已到达
- 登记 expect_tx：把 bytes 加到 transaction count 上，告诉硬件“接下来有多少字节通过 TMA 进来，需要全部搬运完成才算结束”

| 输入 | 含义 | 类型 |
| --- | --- | --- |
| [%0] = bar_addr | mbarrier 对象的 shared memory 地址（32-bit） | .r(u32) |
| %1 = bytes | 本次预计要搬运的字节数量（bytes 必须严格等于 TMA 实际搬运的字节数量—多了死锁，少了线程提前放行会读到半成品） | .r(u32) |

```cpp
mbarrier_arrive_expect_tx(&bar, BM * BK * sizeof(bf16));

// 把"本次将要到达的字节数"作为事务计数登记到 mbarrier
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
    uint32_t bar_addr = __cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                 :: "r"(bar_addr), "r"(bytes));
}

```

**Step 3: 发起 TMA，硬件自动扣减**

`cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
[%0], [%1, {%3, %4}], [%2];`

| 输入 | 含义 | 类型 |
| --- | --- | --- |
| %0 | 要写入的 shared memory 起点（__cvta_generic_to_shared 转换后的 32-bit shared 地址） | .r(u32) |
| %1 | TMA 描述符的 global 指针 | .l(u64) |
| %2 | 关联的 mbarrier 地址：硬件搬完一段就往这个 mbarrier 的 transaction count 扣减 | .r(u32) |
| %3 | TMA 坐标 x（K 维起点，元素为单位，不是字节） | .r(s32) |
| %4 | TMA 坐标 y（M 维起点，元素为单位） | .r(s32) |

硬件每搬完一部分数据，自动把那部分字节数从 bar 的 transaction count 里减掉，当变成 0 之后，触发 phase 翻转。

```cpp
  // 单线程发起一次 TMA tile load：从 (bx, by) tile 拷贝到 smem
  __device__ __forceinline__ void tma_load_2d(
      void const* tma_map, void* smem_ptr, uint64_t* bar,
      int crd_x /*K 方向*/, int crd_y /*M 方向*/)
  {
      uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
      uint32_t bar_addr  = __cvta_generic_to_shared(bar);
      asm volatile(
          "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
          " [%0], [%1, {%3, %4}], [%2];\n"
          :: "r"(smem_addr), "l"(tma_map), "r"(bar_addr),
             "r"(crd_x), "r"(crd_y)
          : "memory");
  }
```

**Step 4: 线程等 phase 翻转**

`mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;`

| 输入 | 含义 | 类型 |
| --- | --- | --- |
| [%0] = bar_addr | mbarrier 对象的 shared memory 地址（32-bit） | .r(u32) |
| %1 = phase | 当前等待的 phase parity（0 或 1） | .r(u32) |

所有线程在这条指令上等，只有当 `arrival count == 0` 且 `transaction count == 0` 时，phase 翻转，`try wait` 返回 true，线程继续。此时可以保证 smem 里的 tile 数据已就绪。

```cpp
// 阻塞等待 mbarrier 翻相位（事务计数清零 = 数据已全部到 smem）
__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
    uint32_t bar_addr = __cvta_generic_to_shared(bar);
    asm volatile(
        "{\n"
        " .reg .pred P;\n"
        "LAB_WAIT:\n"
        " mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        " @P bra DONE;\n"
        " bra LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(bar_addr), "r"(phase));
}
```

### TMA 解决了什么

1. **指令数量**：原本需要 256 * 4 条 `cp.async` ，现在只需要 1 条 TMA 指令。issue 端的搬运压力几乎归零，发射 slot 留给 wgmma。
2. **地址/边界/swizzle：TMA 直接硬件实现，不需要软件计算。**
3. **同步粒度**：从指令组级 `wait_group` 细化为字节级事务计数。

## 总结

| 维度 | LDG | cp.async | TMA |
| --- | --- | --- | --- |
| 引入架构 | Volta+ | Ampere SM_80 | Hopper SM_90 |
| 同/异步 | 同步 | 半异步 | 全异步 |
| 数据通路 | G→Reg→S | G→S 直通 | G→S 直通（硬件 DMA） |
| 发指令的线程 | 每个线程 | 每个线程 | 1 个线程 |
| 地址 / 边界 / swizzle | 软件 | 软件 | 硬件 |
| 同步原语 | __syncthreads | wait_group + __syncthreads | mbarrier 事务计数 |
| 同步粒度 | block 级 | 线程级（指令组） | block 级（字节级） |
| 配套计算指令 | mma.sync | mma.sync (multistage) | wgmma (warp-spec) |