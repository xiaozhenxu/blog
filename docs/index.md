---
template: home.html
hide:
  - navigation
  - toc
  - footer
---

<div class="home-hero" markdown="1">
<div class="hero-copy" markdown="1">
<p class="eyebrow">NOTES ON MODEL INFERENCE</p>

# 理解计算，<br>也理解性能。

<p class="hero-description">记录模型推理与性能优化中的<br class="desktop-break">阅读、实验和思考。</p>

[浏览全部记录 <span aria-hidden="true">↗</span>](#notes){ .text-link }
</div>
<div class="hero-art">
<svg viewBox="0 0 640 280" role="img" aria-labelledby="inference-title" xmlns="http://www.w3.org/2000/svg">
<title id="inference-title">从模型到推理，再到输出的抽象示意图</title>
<defs><pattern id="matrix-grid" width="23" height="23" patternUnits="userSpaceOnUse"><path d="M23 0H0V23" fill="none" stroke="currentColor" stroke-width=".8"/></pattern><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="m1 1 5 3-5 3" fill="none" stroke="currentColor"/></marker></defs>
<g transform="translate(20 46)" stroke="currentColor"><g class="matrix-fill" stroke="none"><path d="M69 0h23v23H69zM23 23h23v23H23zM115 46h23v23h-23zM0 69h23v23H0zM46 92h23v23H46zM92 115h23v23H92z"/></g><rect width="138" height="138" fill="url(#matrix-grid)" stroke-width="1"/></g>
<g stroke="currentColor" fill="none" stroke-width="1"><path d="M183 115h48" marker-end="url(#arrow)"/><path d="M411 115h45" marker-end="url(#arrow)"/><rect x="270" y="49" width="106" height="24"/><rect class="processing-fill" x="270" y="100" width="106" height="24"/><rect x="270" y="151" width="106" height="24"/><path stroke-dasharray="2 5" d="M323 80v14m0 37v14"/>
<rect x="487" y="103" width="20" height="24"/><rect class="matrix-fill" x="517" y="103" width="20" height="24"/><rect x="547" y="103" width="20" height="24"/><rect class="matrix-fill" x="577" y="103" width="20" height="24"/></g>
<g fill="currentColor"><circle cx="611" cy="115" r="1.5"/><circle cx="619" cy="115" r="1.5"/><circle cx="627" cy="115" r="1.5"/></g>
<g class="diagram-label" fill="currentColor" text-anchor="middle"><text x="89" y="230">MODEL</text><text x="323" y="230">INFERENCE</text><text x="555" y="230">OUTPUT</text></g>
</svg>
</div>
</div>

<div class="section-heading" markdown="1">
## 精选记录
[全部记录 ↗](#notes){ .section-link }
</div>

<div class="note-cards" markdown="1">
<div class="note-card" markdown="1">
<p class="eyebrow">01 / INFERENCE</p>

### [模型推理优化](inference/system-optimization/system-optimization.md)

CUDA Graph、计算图变换与性能分析的实践记录。

<div class="card-meta"><span>推理优化 / Profiling</span><span aria-hidden="true">↗</span></div>
</div>
<div class="note-card" markdown="1">
<p class="eyebrow">02 / ATTENTION</p>

### [Flash Attention](operators/flash-attention/flash-attention.md)

分块计算、Online Softmax 与 CUDA 实现笔记。

<div class="card-meta"><span>Attention / 算子优化</span><span aria-hidden="true">↗</span></div>
</div>
<div class="note-card" markdown="1">
<p class="eyebrow">03 / COMPUTE</p>

### [CuTe GEMM](operators/cute-gemm/cute-gemm.md)

Layout、Tiling 与矩阵乘法实现中的细节。

<div class="card-meta"><span>CUDA / CUTLASS</span><span aria-hidden="true">↗</span></div>
</div>
</div>

<div class="section-heading" markdown="1">
## 关注方向
</div>
<div class="topic-grid">
<div><span class="topic-number">01</span><h3>云端推理</h3><p>模型服务、调度与高效生成</p></div>
<div><span class="topic-number">02</span><h3>端侧推理</h3><p>资源约束下的部署与执行</p></div>
<div><span class="topic-number">03</span><h3>推理优化</h3><p>量化、投机解码与 KV Cache</p></div>
</div>

<div class="section-heading" markdown="1">
## 全部记录 {#notes}
<span class="section-caption">随实践逐步补充</span>
</div>
<div class="archive" markdown="1">

- [模型推理优化：以 π₀ 为例](inference/system-optimization/system-optimization.md) <span>推理优化</span>
- [Flash Attention](operators/flash-attention/flash-attention.md) <span>核心算子</span>
- [CuTe GEMM](operators/cute-gemm/cute-gemm.md) <span>核心算子</span>
- [Tensor Cores](fundamentals/tensor-cores/tensor-cores.md) <span>计算基础</span>
- [数据搬运：LDG、cp.async 与 TMA](fundamentals/gpu-data-transfer/gpu-data-transfer.md) <span>计算基础</span>

</div>
<div class="home-about" markdown="1">
这里存放我在模型推理与性能优化方面的阅读、实验和思考，以及一些算子实现与计算基础的笔记。分类用于归档和查找，方便日后回顾和继续探索。
</div>
