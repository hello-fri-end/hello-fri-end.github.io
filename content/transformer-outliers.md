Title: Why do transformers have outliers?
Date: 2026-07-01 
Category: Quantization
Tags: Quantization, Integer Quantization, Deep Learning, Inference, Model Compression, Outliers

Modern Machine Learning models are trained with a large number of parameters, often too large, and this overparameterization is very useful during training as it creates a vast search space for the model to encode rich representations from data into its parameters. However, as it turns out, models do not use all these parameters efficiently, creating a lot of redundancy. 

Training is only a short stage in a model's life cycle and the model spends majority of its time inferencing, where this overparameterization leads to excessive memory usage, slower response times, and increased power utilization. This redundancy is precisely why model compression techniques like pruning, quantization, or low-rank decomposition can often be applied with little loss in accuracy.

Transformer models, however, contain unusually large values concentrated in certain feature dimensions of both weights and activations. These dimensions, known as *outlier channels*, are highly sensitive to model-compression. In this blog post, we will build intuition for why outlier channels emerge in transformer models and what drives their formation.

## How Transformer Models Work
![Transformer Animation](https://substackcdn.com/image/fetch/$s_!0TTv!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F205771d7-cdc8-4fa5-8b5f-de5e916917d5_1920x1080.gif)
*Source: [Mayank Pratap Singh: The need of Attention Mechanism](https://vizuara.substack.com/p/the-need-of-attention-mechanism)*

Transformers convert input into tokens (numbers), where each token is represented as a vector called an embedding. In Vision Transformers (ViTs), images are likewise split into patches, each projected into an embedding. As these representations pass through the model, the goal is to make them increasingly rich by incorporating contextual information from other tokens. Attention blocks play a central role in this contextual refinement.

Each attention block contains multiple attention heads, where different heads capture different kinds of relationships between tokens. Modern transformer models stack many such attention blocks, continually refining token representations by injecting more and more contextual information. This structure is what allows transformers to model sequences so effectively and make strong predictions.

However, this continous refinement of token representations is also where our problem begins.

Think about it for a moment: do all tokens actually contain equal amounts of contextual information or require the same degree of updating? Consider punctuation tokens(. , / ? ), or special tokens such as beginning-of-sequence (BOS) or delimiter tokens (SEP). Similarly, in Vision Transformers (ViTs), many patches may contain very little useful information (e.g background regions like sky, walls, or empty spaces). Even semantically rich tokens may become “good enough” after passing through only a few attention blocks and may not require substantial further updates.

This naturally leads to an important question:

Does the transformer architecture allow token representations to stop updating once they become sufficiently good?

## The Softmax Problem

Recall how attention is computed. Each token embedding is first projected into **queries**, **keys**, and **values**. The query of each token is then matched against all keys via dot products to produce similarity scores. These scores are normalized using a softmax, yielding attention weights (values between 0 and 1) that determine how much each token should contribute to the output. Finally, these weights are used to compute a weighted sum of the value vectors, producing the updated token representations.
![Scaled Dot Product Attention](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*zD2GD96q_v0kJ2vDx3KxeA.png)
*Source: [Bobby Cheng: The Why, What and Where of Transformers](https://medium.com/@bobbycxy/the-why-what-and-where-of-transformers-810ba1763470)*

Recall that the weights need to sum up to 1 (because of how softmax works). That's the heart of the problem. The attention layer cannot simply "do nothing". It must assign some amount of importance somewhere.  
You might imagine a simple workaround: a token just focuses entirely on itself and ignores everything else. Unfortunately, this wouldn't work, because the attention output is always combined back with the original representation through the *residual connection*.


## The Residual Connection Problem

Recall that in GPT-style models (pre-attention layer norm):

$$
x' = x + \mathrm{Attn}(\mathrm{LN}(x))
$$

![Residual Connection](/assets/images/residual.png)

The "no-update" in this structure means attention does not need to preserve \( x \). It only needs to satisfy:

$$
\mathrm{Attn}(\mathrm{LN}(x)) \approx 0
$$

so that:

$$
x' = x
$$

So, how does it work?

## The Hack that Transformers Use

Recall that:

$$
\mathrm{Attn}_i = \sum_j \alpha_{ij} \cdot (V x_j)
$$

Where:

- $\alpha_{ij}$: attention weight from token \(i\) to token \(j\).
- $x_j$: hidden representation. 
- $V$: value projection matrix.


We want:

$$
\mathrm{Attn}_i \approx 0
$$


Since the alphas cannot be all zeros, transformer models tend to assign large weights for tokens which have near-zero value vectors. These are generally tokens with low-semantic content e.g BOS tokens or delimiter tokens.

So, for certain tokens, if: 

$$
V x_j \approx 0 \quad \text{(or very small norm)}
$$

And if attention concentrates on such a token:

$$
\alpha_{ij} \approx 1
$$

then:

$$
\mathrm{Attn}_i \approx V x_j \approx 0
$$


The authors of the ICLR paper [ Efficient Streaming Language Models with Attention Sinks ](https://arxiv.org/pdf/2309.17453) further found out that these are generally initial tokens & called them "Attention Sinks". They hypothesized that, because autoregressive transformers allow every token to attend to earlier tokens, the initial tokens (e.g BOS token) are visible to the entire sequence and therefore become natural accumulation points for excess attention mass. As a result, these early tokens frequently emerge as stable **attention sinks** across many transformer models.

But, what does all of this have to do with outliers?

## The Chain Reaction
We've established that for a token to skip its update, attention must concentrate almost entirely on a token with a near-zero value projection. This requires softmax to produce an almost one-hot distribution.

But how does softmax produce an output close to 1 for one entry and close to 0 for the rest?

From the softmax equation, it is easy to see that this requires the input logits to have a very high dynamic range. Something like the middle row in the following example:

![Softmax](/assets/images/softmax.png)

Mathematically, we want:

$$
q_i \cdot k_j \gg q_i \cdot k_t \quad \text{for } j \ne t
$$

where:

- $q_i$: query vector for token \(i\)

- $k_j$: key vector for token \(j\)

- . : dot product

Intuitively, a token that wants to suppress its update must match very strongly with an attention sink token while matching weakly with all other tokens.

---

From the definitions:

$$
q_i = W_q x_i, \quad k_j = W_k x_j, \quad k_t = W_k x_t
$$

(Subscripts \(i, j, t\) index tokens, and \(x\)'s come from LayerNorm.)

---

Substituting:

$$
(W_q x_i) \cdot (W_k x_j) \gg (W_q x_i) \cdot (W_k x_t)
$$

---

Using the dot product identity:

$$
a \cdot b = \|a\| \, \|b\| \cos \theta
$$

we get:

$$
\|W_q x_i\| \, \|W_k x_j\| \cos \theta_{ij}
\gg
\|W_q x_i\| \, \|W_k x_t\| \cos \theta_{it}
$$

---

Canceling the common term $\|W_q x_i\|$:

$$
\|W_k x_j\| \cos \theta_{ij}
\gg
\|W_k x_t\| \cos \theta_{it}
$$

This inequality can be satisfied along two axes: by making the left-hand side large, or by keeping the right-hand side small. To understand what each requires, we need to answer two questions: what does LayerNorm do to token representations & how to interpret a matrix-vector product?

### Normalization
Recall that LayerNorm normalizes each token using the mean and standard 
deviation computed across its embedding dimension, and then applies a learned 
per-dimension scale and shift parameterized by $\gamma$ and $\beta$, respectively:

$$
x_k \rightarrow \gamma_k \cdot \frac{x_k - \mu}{\sigma} + \beta_k
$$

The squared norm of the output without $\gamma, \beta$ is:

$$
\left\|\frac{x - \mu}{\sigma}\right\|^2
= \frac{1}{\sigma^2} \sum_k (x_k - \mu)^2
= d
$$

since

$$
\sigma^2 = \frac{1}{d} \sum_k (x_k - \mu)^2
$$

by definition.

So every token after LayerNorm has norm exactly $\sqrt{d}$, i.e they all lie on a 
hypersphere of radius $\sqrt{d}$. 
With learnable $\gamma$, all tokens still lie on the same hypersphere but the radius gets scaled by $\|\gamma\|$. $\beta$ slightly perturbs this, but its effect is typically small & the key point holds; magnitude differences across tokens are erased.

We cannot make $\|W_k x_j\|$ large simply because the sink token's representation has a large norm going in. Any difference in output norm must therefore come entirely from directional differences. Let's unpack this next.

### What a linear map does to a vector

A linear transformation

$$
x \mapsto Wx
$$

can **always** be decomposed (via SVD) into three steps:

$$
W = U S V^\top
$$

where:

- $V^\top \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$ is an orthogonal rotation  
  (it changes directions but not lengths or angles).

- $S$ is a diagonal matrix of non-negative singular values

  $$
  \sigma_1 \ge \sigma_2 \ge \cdots \ge 0
  $$

  It stretches/compresses each axis independently.

- $U \in \mathbb{R}^{d_{\text{out}} \times d_{\text{out}}}$ is a second rotation, giving the final orientation.

Thus, applying $W$ to a vector $x$ consists of:

1. **Rotate** $x$ by $V^\top$  
   This re-expresses $x$ in the “principal” coordinate system of $W$.

2. **Scale** each coordinate by the corresponding singular value $\sigma_i$  
   If $x$ happens to align with a singular vector with a large $\sigma$, the output norm becomes much larger than the input norm. If it aligns with a small $\sigma$ (or is orthogonal to large ones), the output becomes tiny.

3. **Rotate** the result by $U$  
   This gives the final output direction.

### Satisfying the inequality
With both pieces in place, the dominance condition becomes a statement about directions:

- Left-hand side: For $\|W_k x_j\| \cos \theta_{ij}$ to be large, two things must hold simultaneously. First, the sink token's representation $x_j$​ must align strongly with the high-singular-value directions of $W_k$, so that the key vector $k_j$ =  $W_k x_j$​ has large norm. Second, the query $q_i$​ must point in the same direction as $k_j$​ for $\cos \theta_{ij}$ to be large. Since $k_j$ is the output of $W_k$​, its direction is determined by $W_k$'s dominant singular structure. For $q_i$​ to reliably land in that same direction, $W_q$​ must strongly amplify it too. This is the core constraint: $W_q$​ and $W_k$​ must develop overlapping dominant singular subspaces, i.e they must agree on which directions to amplify.

- Right-hand side: For $q_i​⋅k_t$​ to remain small for all ordinary tokens $k_t$, those tokens' key vectors must not overlap with those directions. Therefore, the model achieves this by reserving a small set of specialized high-magnitude directions exclusively for the no-op mechanism.

Taken together, both conditions force the model to concentrate representational energy into a small number of disproportionately amplified directions. In weight space, this shows up as a few rows or columns of $W_q$​ and $W_k$ with anomalously large magnitude. In activation space, when inputs pass through these weight matrices, a small number of feature dimensions carry values far larger than the rest.
These are the outlier channels. But how do they form?

## Training Dynamics:
Once outlier channels exist, training actively reinforces them. Softmax with a near-one-hot distribution has gradients concentrated almost entirely on the dominant logit — the one produced by the outlier dimensions. The gradient signal flowing back into $W_q$​ and $W_k$​ is therefore concentrated in precisely those outlier directions, making them stronger with each update.

LayerNorm compounds this. Because it suppresses global scale and equalizes norms across tokens, upstream MLPs must produce outputs large enough that the outlier signal survives normalization and generates sufficient dynamic range for softmax. This pushes earlier layers to amplify outlier dimensions further.

The result is a self-reinforcing loop: outlier channels produce sharp softmax distributions, sharp distributions concentrate gradients back into outlier channels, and the cycle repeats across training.

 Therefore, outliers are not isolated phenomena; they emerge systematically and are interconnected across multiple dimensions within the model. The ICLR paper, [Sytematic Outliers in Large Language Models](https://arxiv.org/pdf/2502.06415) studied how weight outliers give rise to activation outliers, which in turn propagate into attention outliers for Llama 2 7B.

![Activation Outliers from Weight Outliers](/assets/images/activation_outlier_from_weight_outliers.png)

![Attention Outliers from Activation Outliers](/assets/images/attention_outliers_from_activation_outliers.png)

*Source: Systematic Outliers in Large Language Models*

## Conclusion
Some Summary

## References & Further Reading
1. [Quantizable Transformers: Removing Outliers by
Helping Attention Heads Do Nothing](https://arxiv.org/pdf/2306.12929): The first one to discover that outliers emerge because softmax output needs to sum to 1. The authors also propose a "Clipped Softmax" & a "Gated Attention" function which in their experiments prevents the formation of outliers. Larger adoption remains to be unseen though.

2. [Attention Is Off By One](https://www.evanmiller.org/attention-is-off-by-one.html): Evan Miller read the above paper & suggested a simple tweak in the softmax formula: adding a 1 in the denominator so that softmax doesn't necessarily need to sum up to 1. This gained a lot traction when it was released but again it's not used by anyone in practice.

3. [Systematic Outliers in Large Language Models](https://arxiv.org/pdf/2502.06415): The authors did a systematic analysis of outliers by defining weigtht outliers, activation outliers and attention outliers & studied how they give rise to each other. They also study various different attention variants to find out how to prevent the formation of outliers & discover that a learnable scaling factor Sc(x) that dynamically adjusts the attention weights (Explicit Context-Aware Scaling) doesn't lead to the formation of outliers. Again, larger adoption remains to be unseen.

4. [Attention is not all you need](https://arxiv.org/pdf/2103.03404): Not directly related but since the post discusses a subtle limitation of the standard attention mechanism, you'll find this one interesting. The authors essentially show that you just have attention layers & no mlps or residuals in our model, the output converges with a cubic rate to a rank one matrix with identical rows. The MLPs & Residual Connections fight against this. So, the residual connection doesn't just help with gradient flow but also prevents a reprentational collapse.
