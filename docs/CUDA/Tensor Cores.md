# Tensor Cores

## 概述

Tensor Core 是 Nvidia 自 **Volta 架构，即 CUDA 9.0 版本**引入的专门用于**做矩阵乘加的运算单元**。这里举个典型的例子，V100 拥有 640 个 Tensor Core，每块 SM 上拥有 8 个 Tensor Core，因此可以提供 **125 TFLOPS** 的算力。

每个 Tensor Core **在每个时钟周期内可以处理 4\*4\*4 的矩阵乘加运算，即 64 次 FMA**。如下图所表示的运算 $D=A*B+C$，其中 A 和 B 是 **FP16** 类型的矩阵，而累加矩阵 C 和 D 为 **FP16** 或者 **FP32** 类型。

![Tensor Cores 图示](images/Tensor%20Cores/image.png)

![示例图](images/Tensor%20Cores/image%201.png)

一个完整的 **Warp 会并发使用多个 TensorCore**，Warp 内的线程协作会提供一个更大的 **16\*16\*16** 的矩阵运算。我们可以通过调用 **C++ WMMA API** 来实现这些 Warp 级矩阵运算。

## 在 cuBLAS 中使用 Tensor Core

### 代码示例

```cpp
// First, create a cuBLAS handle:
cublasStatus_t cublasStat = cublasCreate(&handle);
 
// Set the math mode to allow cuBLAS to use Tensor Cores:
cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
 
// Allocate and initialize your matrices (only the A matrix is shown):
size_t matrixSizeA = (size_t)rowsA * colsA;
T_ELEM_IN **devPtrA = 0;
 
cudaMalloc((void**)&devPtrA[0], matrixSizeA * sizeof(devPtrA[0][0]));
T_ELEM_IN A  = (T_ELEM_IN *)malloc(matrixSizeA * sizeof(A[0]));
 
memset( A, 0xFF, matrixSizeA* sizeof(A[0]));
status1 = cublasSetMatrix(rowsA, colsA, sizeof(A[0]), A, rowsA, devPtrA[i], rowsA);
 
// ... allocate and initialize B and C matrices (not shown) ...
 
// Invoke the GEMM, ensuring k, lda, ldb, and ldc are all multiples of 8, 
// and m is a multiple of 4:
cublasStat = cublasGemmEx(handle, transa, transb, m, n, k, alpha,
                          A, CUDA_R_16F, lda,
                          B, CUDA_R_16F, ldb,
                          beta, C, CUDA_R_16F, ldc, CUDA_R_32F, algo);
```

!!! tip "使用规则"

    - 只有 GEMM 支持使用 Tensor Core
    - 计算模式必须设置为 `CUBLAS_TENSOR_OP_MATH`
    - `k`、`lda`、`ldb`和`ldc`必须是 8 的倍数；`m`必须是 4 的倍数。Tensor Core 数学例程以 8 个值为一步跨越输入数据，因此矩阵的维度必须是 8 的倍数
    - 矩阵的输入和输出数据类型必须是半精度或单精度
    - 不满足上述规则的 GEMM 将回退到非 Tensor Core 实现

## 在 CUDA C++ 中使用 Tensor Core

CUDA 9.0 通过 `nvcuda::wmma` 命名空间下的一组函数和类型，提供了对 Tensor Core 的支持。下面是一个简单的代码示例。

### 头文件和命名空间

```cpp
#include <mma.h>
using namespace nvcuda;
```

### 声明和初始化

有效的策略是让单个 Warp(线程束) 负责输出矩阵中的一个 16*16 的区域，通过设置合理的 Grid 和 Thread Block，可以**将 Warp 有效地平铺覆盖在二维输出矩阵**上。

```cpp
// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;
 
__global__ void wmma_example(half *a, half *b, float *c, 
                             int M, int N, int K, 
                             float alpha, float beta) 
{
 
    // Leading dimensions. Packed with no transpositions.
    int lda = M;
    int ldb = K;
    int ldc = M;
     
    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
```

在执行 MMA（矩阵乘加）操作之前，参与运算的矩阵必须先被加载到 **GPU 的寄存器**中。由于 WMMA 是 Warp 级操作，这些寄存器分布在 Warp 的各个线程中，每个线程持有整体矩阵的一部分。

在 CUDA 中，`fragment` 是一个模板类型，其模板参数描述了一下内容：

- 该片段所持有的矩阵（`wmma::matrix_a` `wmma::matrix_b` `wmma::accumulator`）
- WMMA 操作的矩阵 shape
- 矩阵数据类型
- 对于`wmma::matrix_a` `wmma::matrix_b` ，需要说明是**行主序**还是**列主序**（最后一个参数可以实现矩阵的**转置**操作）

```cpp
// Declare the fragments
wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
// fill the accumulator fragment with zeros.
wmma::fill_fragment(acc_frag, 0.0f);
```

!!! tip "`fragment` 说明"

    **`fragment`** 在物理上是**分布存储在一个 Warp 32 个线程的寄存器文件**上，它可以通过 `wmma::load_matrix_sync` 指令，将数据从**显存（Global Memory）或共享内存（shared memory）**加载到寄存器中。这个过程是硬件自动分发的。

    - 一个 Warp 32 个线程同时调用`wmma::load_matrix_sync`指令，指向的是同一块数据源地址；
    - GPU 硬件读取内存块中的数据，并根据不透明的规则，将这些数据拆分并投入 32 个线程的寄存器中，每个线程获取**`fragment`的一部分数据。**

### 内部循环

该 GEMM 的策略是让每个 Warp 计算输出矩阵的一个 Tile。为此，需要沿着 A 矩阵的行和 B 矩阵的列进行循环，最终生成一个 M*N 的输出块。

**TODO: Leading Dimension（主维度）**

```cpp
// Loop over the K-dimension
for (int i = 0; i < K; i += WMMA_K) {
    int aRow = warpM * WMMA_M;
    int aCol = i;
    int bRow = i;
    int bCol = warpN * WMMA_N;
     
    // Bounds checking
    if (aRow < M && aCol < K && bRow < K && bCol < N) {
        // Load the inputs
        wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
        wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
 
        // Perform the matrix multiplication
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
}
```

## 参考文档

[Programming Tensor Cores in CUDA 9](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)