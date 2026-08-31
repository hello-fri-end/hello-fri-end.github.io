title: Early Breakthroughs in Transformer Quantization
date: 2026-07-16 
category: Quantization
Tags: Quantization, Integer Quantization, Deep Learning, Inference, Model Compression, Outliers


In my [previous blogpost](https://hello-fri-end.github.io/2026/07/why-do-transformers-have-outliers/), we discussed in detail the outlier problem in transformers. We realized that outliers are not just random noise - they encode important directional information that lets the model selectively skip updating token representations. But the same property makes transformers notoriously difficult to quantize. In this post, we'll discuss why the presence of outliers makes quantization difficult and some early techniques that made quantizing large transformer models possible. 

## **How Outliers Break Quantization**
Good quantization is a balancing act between two errors: rounding error (too many values crammed into too few grid points) and clipping error (values that fall outside the grid entirely). Outliers force you into a lose-lose situation. Include them, and your scaling factor stretches to accommodate their extreme magnitude i.e, the grid becomes so wide that normal values all collapse into the same few bins and rounding error explodes. Clip them, and you incur high clipping error. Moreover, we know now that outliers carry important information & clipping them would significantly affect the model performance.
![Rounding Vs Clipping Error Tradeoff](/assets/images/quantization_tradeoff.png)


Tim Dettmers et al. made exactly this observation in their 2022 LLM.int8() paper. As models scale, a phase transition occurs: large outlier features suddenly emerge across all transformer layers, and naive INT8 quantization completely breaks down at precisely that point.

![LLM.int8()](/assets/images/llmint8.png)
*Source: [LLM.int8(): 8-bit Matrix Multiplication
for Transformers at Scale](https://arxiv.org/pdf/2208.07339)*

From the paper:
> "Setting these outlier feature dimensions to zero decreases top-1 attention softmax probability mass by more than 20% and degrades validation perplexity by 600–1000% despite them only making up about 0.1% of all input features. In contrast, removing the same amount of random features decreases the probability by a maximum of 0.3% and degrades perplexity by about 0.1%"


So, how did LLM.int8() deal with this problem?

## **LLM.int8(): The first to enable 8 bit quantization**
Dettmers' approach was simple and elegant: outliers are clearly doing something important, so keep them in full precision and quantize everything else more carefully using vector-wise quantization. So, what is vector-wise quantization, and how did they find the outliers in the first place?

### **Vector-Wise Quantization**

Matrix multiplication

$$
Y = XW
$$

can be viewed as a collection of independent inner products. Each output element is given by

$$
Y_{ij} = X_{i,:} \cdot W_{:,j},
$$

where $X_{i,:}$ denotes the $i$-th row of the input matrix and $W_{:,j}$ denotes the $j$-th column of the weight matrix.

In **vector-wise quantization**, each input row and each weight column receive their own scaling constants. Let

$$
s^x_i = \max_k |X_{ik}|,
\qquad
s^w_j = \max_k |W_{kj}|.
$$

Using AbsMax quantization, the quantized vectors are

$$
\hat{X}_{i,:} = \mathrm{round}\!\left(\frac{X_{i,:}}{s^x_i}\right),
\qquad
\hat{W}_{:,j} = \mathrm{round}\!\left(\frac{W_{:,j}}{s^w_j}\right).
$$

The inner product is then computed in INT8 as

$$
\hat{Y}_{ij}
=
\hat{X}_{i,:}
\cdot
\hat{W}_{:,j}
$$

To recover the floating-point result, the accumulated value is dequantized using the corresponding scaling factors:

$$
Y_{ij}
\approx
s^x_i s^w_j \hat{Y}_{ij}.
$$

Equivalently,

$$
Y_{ij}
\approx
\left(
s^x_i \hat{X}_{i,:}
\right)
\cdot
\left(
s^w_j \hat{W}_{:,j}
\right).
$$

For the full matrix multiplication, let

$$
s_x =
\begin{bmatrix}
s^x_1 \\
s^x_2 \\
\vdots \\
s^x_m
\end{bmatrix},
\qquad
s_w =
\begin{bmatrix}
s^w_1 &
s^w_2 &
\cdots &
s^w_n
\end{bmatrix}.
$$

The matrix of dequantization factors is simply the outer product

$$
S = s_x s_w,
$$

and the final output can be written as

$$
Y
\approx
\hat{Y}
\odot
(s_x s_w),
$$

where $\odot$ denotes element-wise multiplication.

Thus, each inner product receives its own normalization factor, allowing the quantizer to adapt to the magnitude of individual activation vectors and weight vectors rather than using a single scale for the entire matrix.

#### **Detecting Outliers**
The criteria for detecting outliers from the paper:

> "the magnitude of the feature is at least 6.0, affects at least 25% of layers, and affects at least 6% of the sequence dimensions"

The paper empirically found these thresholds to work well across a wide range of models. Once the outlier columns are identified, they're extracted and kept in fp16 - more on that in the next section.

#### **BitsAndBytes**

The LLM.int8() paper was integrated into the [BitsandBytes](https://github.com/bitsandbytes-foundation/bitsandbytes) library & the huggingface ecosystem. In the actual implementation, the weights are row-wise statically quantized to int8 when they're loaded, i.e if you set the `load_in_8bit` flag to `True`. During the forward pass, the library dynamically scans the activation hidden states for each linear layer column-wise and checks wheter any value in the columns exceeds the `threshold = 6.0`. Once these outlier columns are found, they're kept in fp16 while the rest are quantized to int8. The corresponding weight rows are dequantized back to fp16 and the computation happens as follows:

> X [:, normal_cols]  @  W^T [normal_cols, :]    → int8 x int8

> X [:, outlier_cols] @  W^T [outlier_cols, :]   → fp16 x fp16

So, the dot product is partitioned into two sums & the result of the Matrix Multiplication is the sum of the two partial sums. 


### **Limitations of BitsandBytes**
While LLM.int8() successfully quantized the linear layers of transformer models through vector-wise quantization and mixed-precision decomposition, attention computation — O(n²) in sequence length, and the most expensive operation remained in fp16. Moreover, dynamically scanning activations and decomposing the matmul at runtime added overhead, meaning it offered no inference speedup. Its real contribution was memory: it made loading models like OPT-175B on a single A100 GPU feasible for the first time.

LLM.int8() showed that outliers matter enough to justify keeping them in FP16. However, full-precision retention for even a tiny fraction of weights makes efficient kernel implementation difficult, and buys no compute speedup. What if we could survive quantizing them, just protected differently?

Enter AWQ
## **Activation Aware Quantization**
AWQ came from the MIT Han Lab & claimed near lossless quantization of transformer models to W4A16. Their key insight was that the weight matrices contain these high information carrying channels which they called “salient channels” that constitute about 0.1% - 1% of the parameters and these must be protected during quantization, otherwise accuracy goes for a toss. So, how did they find these salient channels & how did they protect them? Let’s discuss this next.

### **Salient Channels**
They realized that selecting salient weights based on input activation distribution and not weight magnitude or norm can significantly improve the quantization performance. For example, from the table below from the paper, keeping about 1% salient channels selected using activation distribution performs significantly better than selecting based on weight distribution. So much so that, selecting based on the latter is almost as good as selecting at random!

![Salient Channels](/assets/images/salient-channels.png)
*Source: AWQ*

Their hypothesis for this was that there are specific features in the input that are important for model performance & keeping the corresponding weights that get multiplied with them in FP16 preserves them which contributes to better model performance.

However, they also realized that even though these salient channels constitute only about 0.1% - 1% of the weights, keeping them in FP16 would make the system implementation difficult and potentially not lead to any signifacant speedups. So, how did they protect the weights without keeping them in higher precision?

### **Activation-Aware Scaling and Group-Wise Quantization**

Consider a linear layer with full-precision output

$$
Y = XW
$$

and its quantized counterpart

$$
Y' = XQ(W).
$$

We can apply a scaling factor $s$ to the weight matrix and its inverse to the input matrix without changing the output:

$$
Y' = \frac{X}{s} \, Q(Ws).
$$

The core intuition: quantization error in the output isn't just about how precisely a weight is rounded, it's about how much that rounding error gets amplified by the activation multiplying it. A rounding error of 0.01 in a channel with activation magnitude 100 causes 100x more output damage than the same error where activations are near 1. So ideally, we want the quantizer to be most precise where activations are largest.
AWQ's idea is to scale up the weights of salient channels before quantization. Within each group of 128 channels, this shifts how the group's shared step size Δ gets allocated. Channels that got scaled up exert more influence over the group's absmax, pulling the quantizer's resolution toward them.

With this motivation, the quantization problem is formulated as finding a scaling factor that minimizes the mean squared error (MSE) between the original output and the quantized output:

$$
\arg\min_s (Y - Y')^2
=
\arg\min_s
\left(
X
\left(
W - \frac{Q(Ws)}{s}
\right)
\right)^2.
$$

### **Choosing the Scaling Factor**
The authors define the scaling factor as

$$
s =
\frac{s_x^\alpha}
{\sqrt{s_{\max}s_{\min}}},
$$

where:

- $s_x$ is the mean activation computed over a set of calibration samples. It has shape `(in_features)`.
- $\sqrt{s_{\max}s_{\min}}$ is a geometric mean normalization factor that ensures scaling is always a redistribution. The most salient channels are scaled up by exactly as much as the least salient channels are scaled down.

<div class="paper-note">
    <strong>Note.</strong>
    <p>
    Neither detail comes from the paper itself. \(s_x\) as a mean (not a max) and the geometric-mean normalization are both implementation choices visible only in the AutoAWQ code.
    </p>
</div>

- $\alpha$ controls the strength of scaling:
    - $\alpha = 0$ implies no scaling.
    - $\alpha = 1$ implies maximum scaling.

<div class="paper-note">
    <strong>Note.</strong>
    <p>
There is a single \(\alpha\) per weight matrix, so it must be chosen carefully. We want to obtain the benefits of upscaling salient channels without excessively downscaling non-salient channels.
    </p>
</div>

The precise effect of scaling depends on the full distribution of channels within each group: how many salient channels there are, their relative magnitudes, and how they interact to determine the group max. There's no clean closed-form for this - which is exactly why AWQ uses a grid search over α rather than deriving an optimal scale analytically. The activation-aware formulation of s ensures the search is parameterized in the right direction: channels with larger activations get larger scales, so the quantizer is nudged toward protecting the outputs that matter most.


<div class="algo-box" markdown="1">
<div class="algo-title">
<span>Algorithm</span>
<span>AWQ</span>
</div>
<div class="algo-body" markdown="1">

1. Run calibration samples and compute $s_x$ for each linear layer input $X$.
    - $s_x$ has shape `(in_features)`.

2. For each layer, perform a grid search over $\alpha \in [0,1]$ and find the value that minimizes
    $$
    \left(
    X
    \left(
    W - \frac{Q(Ws)}{s}
    \right)
    \right)^2.
    $$

3. Absorb the resulting scaling factors into adjacent layer weights.

4. Run group-wise quantization (group size = 128).

</div>
</div>

AWQ reduces weight memory by 4x, which matters most during the memory-bound decode phase of generation, where the bottleneck is loading weights from memory, not compute. With no runtime overhead (unlike LLM.int8()), this translates directly to real-world speedups: the authors report around 3.2–3.3x average speedups over fp16 across a range of LLMs. The grid search over α is a one-time offline cost. However, computation still happens in fp16, the KV cache remains in fp16, and quantizing very large models is slow due to the per-layer grid search.


Another popular weight-only quantization method, GPTQ, was released around the same time. We'll cover that in the next post.

Let's finally discuss an early approach that unlocked W8A8 quantization for transformers. 

## **SmoothQuant**

SmoothQuant was also developed by the [MIT-HAN Lab](https://hanlab.mit.edu/) and was the first method to make **8-bit activation quantization** practical for large language models. This enabled all matrix multiplications to be performed in INT8 while also allowing the KV cache to be stored in 8-bit precision.

The authors observed that **per-channel activation quantization** almost completely recovers the accuracy of the original model. This finding is consistent with previous studies, such as [Understanding and Overcoming the Challenges of Efficient Transformer Quantization](https://arxiv.org/abs/2109.12948) from Qualcomm.

![Per-Channel Quantization of Activations Preserves Accuracy](/assets/images/perchannelact.png)
*Source: SmoothQuant*


This works because per-channel quantization assigns each activation channel its own scale factor: outlier channels no longer hijack the shared scale and destroy the precision of normal channels. 

If you've read the [first blog post](https://hello-fri-end.github.io/2026/06/integer-quantization-deep-dive/#how-quantization-is-applied-quantization-granularity) in this series, you'll know that per-channel activation quantization is not hardware-friendly. Therefore, if activations are to be quantized to 8 bits efficiently, the quantization must be performed **per tensor**. With that constraint in mind, let's examine how SmoothQuant enables accurate activation quantization.

### **The Smoothing Transformation**

The core idea behind SmoothQuant is to **migrate quantization difficulty from activations to weights** using a mathematically equivalent transformation that can be applied offline.

For a linear layer,

$$
Y = XW
= \left(X \, \mathrm{diag}(s)^{-1}\right)
\left(\mathrm{diag}(s)W\right)
= \hat{X}\hat{W}
$$

The goal is to choose a scaling vector $s$ such that $\hat{X}$ is easier to quantize than $X$.

A simple approach would be to choose each element of $s$ as the maximum value of the corresponding activation channel, scaling all activation values into [0, 1]. This would make per-tensor activation quantization highly effective. However, doing so transfers all quantization difficulty to the weights, making them much harder to quantize accurately.

Instead, SmoothQuant seeks to **balance the quantization difficulty** between activations and weights. The authors propose the following scaling factor for channel $j$:

$$
s_j =
\frac{\max(|X_j|)^\alpha}
     {\max(|W_j|)^{1-\alpha}}
$$

where:

- $\alpha = 1$ pushes all quantization difficulty to the weights.
- $\alpha = 0$ pushes all quantization difficulty to the activations.
- Intermediate values distribute the difficulty between the two.

The authors found empirically that **$\alpha = 0.5$** provides a good balance, enabling both activations and weights to be quantized per tensor with minimal accuracy degradation.

Moreover, this transformation is applied offline: the scaling vector s is absorbed into the weights of the preceding linear layer or the scale parameters of a preceding LayerNorm. At inference time, the smoothed activations X̂ arrive already transformed & there's no runtime overhead.

Overall, SmoothQuant achieves approximately **2× memory reduction** and provides up to **1.56× inference speedup**, according to the authors.

## **Conclusion**
Thank you for reading! I hope the post helped you understand some very popular quantization algorithms for transformers. The following table contains a summary of the details:

| Technique | What's quantized | Granularity | Bitwidth | Computation precision |
|---|---|---|---|---|
| **LLM.int8()** | Weights + activations, except ~0.1% outlier feature dimensions kept in FP16 | Vector-wise (per-row for activations, per-column for weights) | INT8 (regular), FP16 (outliers) | Mixed: INT8 matmul for the bulk, FP16 matmul for the outlier decomposition |
| **AWQ** | Weights only | Per-group (group size 128) | INT4 (W4A16) | FP16: activations and compute stay full precision; only weight storage/bandwidth is reduced |
| **SmoothQuant** | Weights + activations + KV cache (K/V as BMM inputs) | Per-tensor (static, calibrated offline) | INT8 (W8A8) | INT8 for GEMMs and attention BMMs; FP16 kept for lightweight ops (softmax, LayerNorm) |

In the next one, we'll start looking into rounding schemes that go beyond simple round-to-nearest to unlock even better quantization quality. See you there! 👋

## **References & Further Reading**
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
- [BitsAndBytes (LLM.int8() reference implementation)](https://github.com/bitsandbytes-foundation/bitsandbytes)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- [AWQ reference implementation (MIT HAN Lab)](https://github.com/mit-han-lab/llm-awq)
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)
- [SmoothQuant reference implementation (MIT HAN Lab)](https://github.com/mit-han-lab/smoothquant)
- [Understanding and Overcoming the Challenges of Efficient Transformer Quantization](https://arxiv.org/abs/2109.12948)