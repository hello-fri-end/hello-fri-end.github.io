title: Part 2: Adaptive Rounding: Optimal Brain Quantizer (OBQ) 
date: 2026-09-02 
category: Quantization
Tags: Quantization, Integer Quantization, Deep Learning, Optimal Brain Quantizer, Model Compression, OBQ

In Part 1, we looked at why rounding direction matters: a stochastic rounding experiment on a single ResNet-18 layer found assignments that beat round-to-nearest by over 10 points, just by choosing up vs. down more carefully. Then we looked into second order analysis of quantization, and reazlied that the damage is governed entirely by the Hessian, which is very expensive to compute for large models, and Adaround makes it tractable with three simplications: layer-wise, diagonal & constant curvature - this collapses the objective to a simple asymmetric M.S.E between the full-precision network and the quantized layer outputs. However, we also saw that Adaround requires 10k-15k gradient steps per layer to converge for 8 bits and 4 bits respectively.

In this post and the next, we will look at OBQ and GPTQ -- the methods that made second-order adaptive rounding tractable at scale.

## Problem Formulation

Recall the following optimization objective we derived using the Taylor series 
expansion in the AdaRound blogpost:

$$
\arg\min_{\Delta \mathbf{w}^{(\ell)}} \; \mathbb{E}\left[\Delta\mathbf{w}^{(\ell)\top} \cdot \mathbf{H}(\mathbf{w}^{(\ell)}) \cdot \Delta\mathbf{w}^{(\ell)}\right]
$$

where:

- $\Delta \mathbf{w}^{(\ell)} = \hat{\mathbf{w}}^{(\ell)} - \mathbf{w}^{(\ell)}$ is the perturbation introduced by 
  quantizing the weights at layer $\ell$ i.e, the vector of rounding errors
- $\mathbf{H}(\mathbf{w}^{(\ell)})$ is the Hessian of the task loss with respect to the 
  weights at layer $\ell$, capturing how sensitive the loss is to perturbations 
  in each weight and each pair of weights

Minimizing this objective means finding the quantized weights $\hat{\mathbf{w}}^{(\ell)}$ 
whose rounding errors cause the least damage to the loss, weighted by the 
curvature of the loss landscape.

The challenge is that computing the full Hessian $\mathbf{H}(\mathbf{w}^{(\ell)})$ is 
intractable for large layers. Let's now understand how OBQ approximates it it efficiently enough to apply at scale. 

### From Layer Error to Row-wise Weight Errors

Consider a single output:

$$
y = \sum_j w_j x_j
$$

where:

- $w_j$ is the $j$-th weight in a given row
- $x_j$ is the corresponding input activation

Our goal is to preserve $y$ as much as possible after quantization. Notice that 
a single output value depends only on one row of the weight matrix, which means 
**we can treat each row independently**.

Quantizing weight $w_j$ introduces a rounding error:

$$
\Delta w_j = \hat{w}_j - w_j
$$

The resulting output error is:

$$
\Delta y = \hat{y} - y = \sum_j \Delta w_j \, x_j
$$

### Goal

A natural proxy objective is to minimize the expected squared output error:

$$
L = \mathbb{E}\left[(\Delta y)^2\right]
$$

Substituting the expression for $\Delta y$:

$$
L = \sum_i \sum_j \Delta w_i \, \Delta w_j \; \mathbb{E}[x_i x_j]
$$

Defining:

$$
\mathbf{H} = \mathbb{E}[\mathbf{x}\mathbf{x}^\top]
$$

gives:

$$
L = \Delta\mathbf{w}^\top \mathbf{H} \, \Delta\mathbf{w}
$$

Comparing this with the Taylor-series objective above, we can see they have 
identical structure — $\Delta\mathbf{w}^\top \mathbf{H} \Delta\mathbf{w}$ — which means the 
input covariance matrix $\mathbf{H} = \mathbb{E}[\mathbf{x}\mathbf{x}^\top]$ is the 
**tractable proxy for the true Hessian**. Rather than computing second 
derivatives of the task loss (expensive and requiring backprop through the 
entire network), we approximate the Hessian using only the layer's input 
activations, which are cheap to collect with a small calibration dataset.

 With that background, let's unpack OBQ step by step.

## OBQ: Motivation
At its core, OBQ asks the following question:

> Quantizing weight $w_j$ introduces a fixed rounding error $\Delta w_j$. 
> How should the remaining weights in the current row be adjusted to absorb this error?

Consider a simple example with three weights:

$$
\mathbf{w} = [w_1, w_2, w_3].
$$

Suppose we quantize $w_1$, introducing a fixed rounding error:

$$
\Delta w_1 = \hat{w}_1 - w_1.
$$

We would like to compensate by adjusting the remaining weights:

$$
w_2 \rightarrow w_2 + \Delta w_2
$$

$$
w_3 \rightarrow w_3 + \Delta w_3
$$

The challenge is to find $\Delta w_2$ and $\Delta w_3$ that minimize the output damage.

### Recall the Proxy Loss

From the previous section, the proxy loss has the same structure as the 
Taylor-series objective, with the input covariance as the Hessian:

$$
L = \Delta\mathbf{w}^\top \mathbf{H} \, \Delta\mathbf{w}
$$

where $\Delta\mathbf{w} = [\Delta w_1, \Delta w_2, \Delta w_3]^\top$.

Expanding the quadratic form:

$$
L =
H_{11}\Delta w_1^2
+ 2H_{12}\Delta w_1 \Delta w_2
+ 2H_{13}\Delta w_1 \Delta w_3
+ H_{22}\Delta w_2^2
+ 2H_{23}\Delta w_2 \Delta w_3
+ H_{33}\Delta w_3^2.
$$

Since $\Delta w_1$ is fixed (determined when $w_1$ was quantized), we minimize 
$L$ with respect to $\Delta w_2$ and $\Delta w_3$.


### Taking Partial Derivatives

Differentiating with respect to $\Delta w_2$:

$$
\frac{\partial L}{\partial \Delta w_2}
= 2H_{12}\Delta w_1 + 2H_{22}\Delta w_2 + 2H_{23}\Delta w_3.
$$

Setting to zero:

$$
H_{22}\Delta w_2 + H_{23}\Delta w_3 = -H_{12}\Delta w_1.
$$

Differentiating with respect to $\Delta w_3$:

$$
\frac{\partial L}{\partial \Delta w_3}
= 2H_{13}\Delta w_1 + 2H_{23}\Delta w_2 + 2H_{33}\Delta w_3.
$$

Setting to zero:

$$
H_{23}\Delta w_2 + H_{33}\Delta w_3 = -H_{13}\Delta w_1.
$$


### Linear System

Collecting both equations:

$$
\begin{bmatrix}
H_{22} & H_{23} \\
H_{23} & H_{33}
\end{bmatrix}
\begin{bmatrix}
\Delta w_2 \\
\Delta w_3
\end{bmatrix}
=
-\Delta w_1
\begin{bmatrix}
H_{12} \\
H_{13}
\end{bmatrix}.
$$

Solving:

$$
\begin{bmatrix}
\Delta w_2 \\
\Delta w_3
\end{bmatrix}
=
-\Delta w_1
\begin{bmatrix}
H_{22} & H_{23} \\
H_{23} & H_{33}
\end{bmatrix}^{-1}
\begin{bmatrix}
H_{12} \\
H_{13}
\end{bmatrix}
\tag{1}
$$

This tells us how the remaining weights should be adjusted to compensate for 
the error introduced when quantizing $w_1$.

Notice that $\Delta w_2$ and $\Delta w_3$ cannot be computed independently. 
The optimal adjustment to $w_2$ depends on the adjustment applied to $w_3$, 
and vice versa. This happens because the corresponding input features are 
correlated, which is captured by the off-diagonal term $H_{23}$.

In other words, the remaining weights must **coordinate** their adjustments 
to minimize the overall output damage.

### The OBS / GPTQ Trick
A naive implementation of the above procedure would look something like this:

<div class="algo-box" markdown="1">
<div class="algo-title">
<span>Naive Algorithm</span>
<span>OBQ</span>
</div>
<div class="algo-body" markdown="1">
1. Quantize $w_1$.
2. Construct the Hessian corresponding to the remaining weights.
3. Invert that Hessian.
4. Solve for the optimal error compensation.
5. Quantize $w_2$.
6. Construct a new Hessian for the still-unquantized weights.
7. Invert it again.
8. Repeat until all weights have been quantized.
</div>
</div>

The problem with this implementation is that matrix inversion is expensive. Inverting an $n \times n$ matrix typically costs $O(n^3)$, which quickly becomes prohibitive for modern neural network layers containing thousands of weights.

The key insight behind OBQ, later leveraged by GPTQ, is that these repeated inversions are unnecessary. Instead of recomputing the inverse Hessian after every quantization step, OBQ proves that the inverse corresponding to the remaining weights can be obtained directly from the original inverse Hessian by subtracting an outer-product from it to perfectly simulate having removed one weight. 

Suppose weight $p$ has just been quantized. Let $H^{-1}$ denote the inverse Hessian before removing that weight. The inverse Hessian for the remaining (unquantized) weights is

$$
H^{-1}_{-p}
=
H^{-1}
-
\frac{1}
{\left(H^{-1}\right)_{pp}}
H^{-1}_{:,p}
H^{-1}_{p,:},
$$

where

- $H^{-1}_{:,p}$ is the $p$-th column of the inverse Hessian,
- $H^{-1}_{p,:}$ is the $p$-th row of the inverse Hessian, and
- $\left(H^{-1}\right)_{pp}$ is the diagonal entry corresponding to the quantized weight.

The **outer product** $H^{-1}_{:,p} H^{-1}_{p,:}$ costs only $O(n^2)$, a significant improvement over the $O(n^3)$ cost of performing a fresh matrix inversion.

This update can be interpreted as a form of **Gaussian elimination**. Rather than explicitly removing the quantized weight and reinverting the reduced Hessian, we perform elementary row and column operations on the inverse Hessian that eliminate the influence of weight $p$. The resulting matrix is exactly the inverse Hessian we would have obtained had we removed that weight and recomputed the inverse from scratch.

#### Generalized Weight Update

Equation (1) showed how to optimally redistribute the quantization error 
from a single weight to the remaining weights in our three-weight example. 
The authors generalized this result to an arbitrary number of weights.

Suppose weight row $w_p$ has just been quantized, introducing a rounding error:

$$
\Delta w_p = \hat{w}_p - w_p.
$$

The optimal adjustment to all remaining (unquantized) weights is:

$$
\Delta w_{\text{remaining}}
=
\Delta w_p
\frac{
\left(\mathbf{H}^{-1}\right)_{p,\ \text{remaining}}
}{
\left(\mathbf{H}^{-1}\right)_{pp}
}.
$$

The remaining weights are then updated as:

$$
w_{\text{remaining}}
\leftarrow
w_{\text{remaining}}
+
\Delta w_p
\frac{
\left(\mathbf{H}^{-1}\right)_{p,\ \text{remaining}}
}{
\left(\mathbf{H}^{-1}\right)_{pp}
}.
$$

### Greedy Search

So far, we've formulated quantization as a layer-wise output reconstruction 
problem, derived the optimal weight compensation rule, and shown how to 
efficiently update the inverse Hessian after each quantization step.

The remaining question is:

> **How do we choose which weight to quantize at each step?**

OBQ answers this with a **greedy search**. At every step, it selects the 
weight whose quantization incurs the smallest increase in the proxy loss.

From the OBQ derivation, quantizing weight $w_p$ increases the loss by:

$$
\Delta L_p = \frac{(\Delta w_p)^2}{(\mathbf{H}^{-1})_{pp}}
$$

where $\Delta w_p = \hat{w}_p - w_p$ is the rounding error introduced by 
quantizing $w_p$. At every iteration, OBQ selects the unquantized weight 
with the smallest $\Delta L_p$.

<div class="algo-box" markdown="1">
<div class="algo-title">
<span>Algorithm</span>
<span>OBQ</span>
</div>
<div class="algo-body" markdown="1">
1. Process the network one layer at a time.
2. Run the calibration dataset through the network and collect input 
   activations $\mathbf{X}$ for the current layer.
3. Compute the proxy Hessian:
   $$\mathbf{H} = \mathbf{X}\mathbf{X}^\top$$
4. Compute $\mathbf{H}^{-1}$.
5. For each row of the weight matrix:
      1. Compute $\Delta L_p = \frac{(\Delta w_p)^2}{(\mathbf{H}^{-1})_{pp}}$ 
         for every unquantized weight.
      2. Quantize the weight with the smallest $\Delta L_p$.
      3. Update remaining weights using the generalized compensation rule.
      4. Update $\mathbf{H}^{-1}$ via the rank-one update.
      5. Repeat until all the weights are quantized.
6. Move to the next layer and repeat.
</div>
</div>


### Limitations of OBQ

OBQ produces excellent quantization quality but does not scale to large 
language models. Let's understand why:

#### 1. Greedy Search Cost
Consider a layer with weight matrix of size $4096 \times 4096$. For a 
single row, OBQ evaluates all 4096 unquantized weights via $\Delta L_p$, 
selects the best, quantizes it, compensates the remaining 4095 weights, 
and updates $\mathbf{H}^{-1}$. This is repeated 4096 times for all the columns. Across 
4096 rows, the total work becomes enormous.

#### 2. Parallelism Across Rows Is Broken by Greedy Search

In principle, each row of the weight matrix can be quantized independently. 
The greedy search breaks this.

Suppose we quantize two rows simultaneously. Row 1's greedy search picks 
column 1 first; row 2's picks column 200. After these decisions, the two 
rows need **different** inverse Hessians i.e, one with row and column 1 
eliminated, the other with row and column 200 eliminated. They can no 
longer share a single $\mathbf{H}^{-1}$.

For a $4096 \times 4096$ layer, one copy of $\mathbf{H}^{-1}$ holds 
$4096^2 \approx 16.8\text{M}$ entries, roughly **64 MB** in float32. 
Quantizing $k$ rows in parallel requires $k$ separate copies, so memory 
scales linearly with the degree of parallelism. At any meaningful scale, 
this is prohibitive.

## Conclusion

Thank you for reading! I hope the post helped demystify OBQ, the foundation on which the very popular technique GPTQ is built. We'll cover GPTQ in the next post. See you there! 

Here's a quick summary of OBQ's design choices:

| Bit-width | Granularity | What's quantized | What's not quantized | Compute precision | Scheme |
|---|---|---|---|---|---|
| 4-bit (typical) | Per-row | Weights only | Activations, KV-cache | FP16 | Asymmetric uniform |


## References and Further Reading

1. [Optimal Brain Compression: A Framework for Accurate Post-Training 
Quantization and Pruning](https://arxiv.org/abs/2208.11580)
2. [Optimal Brain Surgeon and General Network Pruning](https://ieeexplore.ieee.org/document/298572) 
3. [Up or Down? Adaptive Rounding for Post-Training Quantization](https://arxiv.org/abs/2004.10568) 
4. [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained 
Transformers](https://arxiv.org/abs/2210.17323) 