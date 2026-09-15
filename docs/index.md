# 理解计算，也理解性能。

这里记录我在模型推理与性能优化方面的阅读、实验和思考，也存放一些算子实现与计算基础的笔记，方便日后回顾和继续探索。

内容会随着实践逐步补充，分类主要用于归档和查找。

## 推理优化

模型推理中的优化尝试、性能分析与相关资料整理。

- [模型推理优化：以 π₀ 为例](inference/system-optimization/system-optimization.md)：CUDA Graph、计算图变换、Triton 优化与性能分析笔记。

## 核心算子

阅读和实现算子时，对算法、代码及数据布局的记录。

- [Flash Attention](operators/flash-attention/flash-attention.md)：分块计算、Online Softmax 与 CUDA 实现笔记。
- [CuTe GEMM](operators/cute-gemm/cute-gemm.md)：Layout、Tiling、MMA 与矩阵乘法实现笔记。

## 计算基础

实践中涉及的硬件概念、指令与数据搬运机制。

- [Tensor Cores](fundamentals/tensor-cores/tensor-cores.md)：基本概念与 cuBLAS、CUDA C++ 使用记录。
- [数据搬运：LDG、cp.async 与 TMA](fundamentals/gpu-data-transfer/gpu-data-transfer.md)：同步加载、异步拷贝与 TMA 的对比整理。
