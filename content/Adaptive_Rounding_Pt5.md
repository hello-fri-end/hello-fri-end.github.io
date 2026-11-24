title: Part 5: Adaptive Rounding: Blockwise Reconstruction Quantization (BRECQ)
date: 2026-11-25
category: Quantization
Tags: Quantization, Integer Quantization, Deep Learning, Model Compression, Adaround, BRECQ, Blockwise Reconstruction

So far we have looked at adaptive rounding techniques that find the optimal grid point assignment for each weight by minimizing the layer-wise output difference between the floating-point model and the quantized model. One drawback of this approach is that it's biased: it assumes that interactions between layers don't matter and we only need to reconstruct the output of each layer correctly. On the other end of the spectrum, we could reconstruct the output of the final layer, i.e., do a full network-wise reconstruction. But this suffers from high variance (overfitting), since we would be tuning a huge number of parameters against only about a few hundred calibration samples.


The authors of BRECQ argue, and show empirically, that there is a middle ground with a better bias-variance tradeoff: reconstructing at the ***block*** level. A block is a building block of the architecture, e.g., a ResNet residual block. The paper reports BRECQ as the first PTQ method to retain usable accuracy with 2-bit weights on models like ResNet and MobileNetV2. Later methods carry block reconstruction over to transformers and language models, e.g., [QDrop](https://arxiv.org/pdf/2203.05740), [FlexRound](https://arxiv.org/pdf/2306.00317) and [OmniQuant](https://arxiv.org/abs/2308.13137). So, let's understand the fundamental ideas behind BRECQ.

## Problem formulation

In my [AdaRound blogpost](https://hello-fri-end.github.io/2026/08/part-1-adaptive-rounding-adaround/), we saw how the authors made some simplifying assumptions to sidestep the Hessian computation, reducing the layer-wise reconstruction problem to a simple MSE between the full-precision and quantized layer outputs (pre-activations). Specifically, the authors assumed:

1. Layers are mutually independent, so the Hessian is layer-diagonal.
2. The Hessian w.r.t. the pre-activations of a layer is diagonal, i.e., no cross-terms between different output neurons.
3. The diagonal entries are all the same constant $c$, independent of data samples, i.e., $H = cI$.

The authors of BRECQ argue that these assumptions break down at lower bit-widths such as INT2, where the weight perturbation $\Delta w$ is no longer small. They point to the ignored cross-layer dependency as the main culprit.


### Modeling Cross-Layer Dependency

To get rid of the layer-independence assumption, BRECQ starts from the whole network instead of a single layer and asks what the Hessian actually looks like. The key observation is that the loss only reaches the weights *through the network's final output*, and that gives us a clean way to write down cross-layer interactions.

**Notation**

| Symbol | What it is | Size |
|---|---|---|
| $\theta$ | All the weights of all $n$ layers, stacked into one long vector | $d$ |
| $z^{(n)} = f(\theta)$ | The network's final output (e.g., logits) | $m$ |
| $L$ | The task loss. It depends on $\theta$ only through $z^{(n)}$ | scalar |
| $J = \partial z^{(n)} / \partial \theta$ | The Jacobian: how much each output moves when each weight moves | $m \times d$ |
| $H(z^{(n)})$ | The Hessian of the loss w.r.t. the output: how sharply the loss curves around the output | $m \times m$ |

A bar, as in $\bar{H}$, means "averaged over the data".

**Step 1: Differentiate twice.** By the chain rule, the first derivative of the loss w.r.t. one weight sums over every output neuron it can influence:

$$
\frac{\partial L}{\partial \theta_i} = \sum_{k=1}^{m} \frac{\partial L}{\partial z_k^{(n)}} \frac{\partial z_k^{(n)}}{\partial \theta_i}
$$

Differentiating again w.r.t. $\theta_j$ means differentiating a product, so we get two terms:

$$
\frac{\partial^2 L}{\partial \theta_i \partial \theta_j} = \underbrace{\sum_{k} \frac{\partial L}{\partial z_k^{(n)}} \frac{\partial^2 z_k^{(n)}}{\partial \theta_i \partial \theta_j}}_{\text{term 1}} + \underbrace{\sum_{k,l} \frac{\partial z_k^{(n)}}{\partial \theta_i} \frac{\partial^2 L}{\partial z_k^{(n)} \partial z_l^{(n)}} \frac{\partial z_l^{(n)}}{\partial \theta_j}}_{\text{term 2}}
$$

**Step 2: Drop term 1 (Assumption: the model has converged).** Every piece of term 1 is multiplied by $\partial L / \partial z_k^{(n)}$, the gradient of the loss w.r.t. the output. The paper assumes this gradient is $\approx 0$ for a converged model, so term 1 vanishes. (Strictly, convergence only guarantees that the *average* gradient w.r.t. the weights is $\approx 0$; the per-output version holds best when the model fits the data confidently.) Convergence also suggests that the Hessian is [positive-semidefinite (PSD)](https://en.wikipedia.org/wiki/Definite_matrix), i.e., it cannot curve further down. What's left is term 2, which in matrix form is the **Gauss-Newton matrix** ([Botev et al., 2017](https://arxiv.org/abs/1706.03662)):

$$
H(\theta) \approx G(\theta) = J^T H(z^{(n)}) J
$$

In practice, we cannot explicitly compute and store the Jacobian for each input data point in such a raw form. Let's see how to simplify this next:

**Step 3: Move the problem from weight space to output space.** Recall that quantization perturbs the weights by $\Delta\theta$, and the resulting increase in loss using taylor series expansion is:

$$
\mathbb{E}\left[L(\theta + \Delta\theta)\right] - \mathbb{E}\left[L(\theta)\right] \approx \Delta\theta^T \bar{g}(\theta) + \frac{1}{2}\Delta\theta^T \bar{H}(\theta) \Delta\theta
$$

For a converged model the gradient term $\bar{g}$ is $\approx 0$, so what we want to minimize is the quadratic form $\Delta\theta^T \bar{H}(\theta) \Delta\theta$. Plugging in $G(\theta)$ for the Hessian (we'll bring the expectation back at the end) and regrouping:

$$
\Delta\theta^T H(\theta) \Delta\theta \approx \Delta\theta^T J^T H(z^{(n)}) J \Delta\theta = (J\Delta\theta)^T H(z^{(n)}) (J\Delta\theta)
$$

By definition of the Jacobian, $J\Delta\theta$ is the first-order estimate of how much the output moves when every weight is perturbed:

$$
\Delta z^{(n)} = \hat{z}^{(n)} - z^{(n)} \approx J \Delta\theta
$$

**Assumption: the output changes linearly in $\Delta\theta$ (first-order approximation).** Substituting this in, and taking the expectation over data, gives the central result:

$$
\Delta\theta^T \bar{H}(\theta) \Delta\theta \approx \mathbb{E}\left[\Delta z^{(n)T} H(z^{(n)}) \Delta z^{(n)}\right]
$$

### Block Diagonal Assumption
The formulation above corresponds to reconstructing the output of the full network. It accurately captures the second-order error, but as discussed in the introduction, reconstructing at the block level gives a better bias-variance tradeoff.

Formally, suppose layers $k$ through $\ell$ (where $1 \le k < \ell \le n$) form a block. Define the weight vector of that block as

$$
\tilde{\theta} = \text{vec}\left[w^{(k)T}, \ldots, w^{(\ell)T}\right]^T
$$

Applying the same derivation as before, but now to the block's own output $z^{(\ell)}$, the second-order error becomes

$$
\Delta\tilde{\theta}^T \bar{H}(\tilde{\theta}) \Delta\tilde{\theta} \approx \mathbb{E}\left[\Delta z^{(\ell)T} H\left(z^{(\ell)}\right) \Delta z^{(\ell)}\right]
$$

where $\Delta z^{(\ell)}$ is the change in the block's output caused by quantization, and $H(z^{(\ell)})$ is the Hessian of the task loss w.r.t. that output. This amounts to a **block-diagonal Hessian**: it keeps the dependencies between layers *within* a block, but ignores the dependencies *between* blocks. Since it optimizes far fewer parameters at a time than the full network, it also produces less generalization error.
![Various Reconstruction Granularities](/assets/images/reconstruction_granulatires.png)
*Source: [BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction](https://arxiv.org/pdf/2102.05426)*

### Approximating the Pre-Activation Hessian

With block reconstruction, the objective for each block is

$$
\mathbb{E}\left[\Delta z^{(\ell)T} H\left(z^{(\ell)}\right) \Delta z^{(\ell)}\right]
$$

We know $\Delta z^{(\ell)}$ (run the full-precision and quantized block and subtract). What's left is $H(z^{(\ell)})$: how curved the **task loss** is around this block's output. Computing it exactly means backpropagating *second* derivatives through every layer after the block, which is the same computational wall we were trying to avoid.

AdaRound's answer was to assume $H(z^{(\ell)}) = cI$, which turns the objective into plain MSE. That is cheap, but it treats every output element as equally important to the final loss. BRECQ instead uses the **Fisher Information Matrix (FIM)**, a quantity that behaves like a curvature but needs only first derivatives.

#### What is the FIM?

Consider a model that assigns a probability $p_\theta(y|x)$ to each outcome. The **score** is the gradient of the log-probability, $s = \nabla_\theta \log p_\theta(y|x)$. It tells you which direction would make the model explain this data point better. The FIM is the expected outer product of the score with itself:

$$
F(\theta) = \mathbb{E}\left[\nabla_\theta \log p_\theta(y|x)\, \nabla_\theta \log p_\theta(y|x)^T\right]
$$

Two facts make this useful as a curvature stand-in (both with the expectation taken under the model's own distribution). First, the score then has zero mean, so $F$ is simply its covariance: how much the slope of the log-likelihood swings from sample to sample. Second, the same quantity is the negative expected Hessian of the log-likelihood:

$$
F(\theta) = -\mathbb{E}\left[\nabla_\theta^2 \log p_\theta(y|x)\right]
$$

So where the slope swings widely, the log-likelihood is sharply curved, and we get a curvature measure from first derivatives alone.

#### The diagonal assumption

The full FIM is still a matrix with off-diagonal entries. BRECQ keeps only the diagonal, where each entry is a squared gradient that an ordinary backward pass already produces:

$$
F_{kk} = \mathbb{E}\left[\left(\frac{\partial L}{\partial z^{(\ell)}_k}\right)^2\right]
$$

This has the same form as Adam's second-moment estimate (there applied to weight gradients, here to gradients at the block output).

#### The result: a weighted MSE

Plugging the diagonal FIM into the block objective gives

$$
\mathbb{E}\left[\Delta z^{(\ell)T}\, \text{diag}\left(\left(\tfrac{\partial L}{\partial z_1^{(\ell)}}\right)^2, \ldots, \left(\tfrac{\partial L}{\partial z_a^{(\ell)}}\right)^2\right) \Delta z^{(\ell)}\right] = \mathbb{E}\left[\sum_{k} \left(\frac{\partial L}{\partial z_k^{(\ell)}}\right)^2 \left(\Delta z_k^{(\ell)}\right)^2\right]
$$

where $a$ is the number of elements in $z^{(\ell)}$. Each output element's squared error is scaled by how much the task loss cares about that element. Elements the loss is sensitive to are penalized heavily for quantization error, and elements it barely notices are cheap to get wrong.

<div class="paper-note">
    <strong>Note.</strong>
    <p>
        The FIM equals the Hessian of the task loss only if the model's distribution matches the true data distribution. A converged network is plausibly close to this, but never exactly. The BRECQ authors say as much: it's "the best we can do since the pretrained model is converged." Also, plugging in gradients of the loss computed with the true labels gives the <em>empirical</em> Fisher, which can differ from the true Fisher (where labels are sampled from the model).
    </p>
</div>

### Setup

1. **Collect calibration data.** Sample a small calibration set (1024 images in the paper).

2. **Split the network into reconstruction units.** Each building block (e.g., a ResNet residual block) is one unit. Layers that don't belong to any block, such as the first convolution and the final fully connected layer, are reconstructed layer-wise, exactly as in AdaRound.

3. **Compute block inputs and targets.** For the current block:

    - the target output $z$ is obtained by forwarding the calibration data through the original full-precision network;
    - the quantized input $\hat{x}$ is obtained by forwarding the same data through the network where all preceding blocks have already been quantized, so their rounding errors carry forward into this block.

4. **Cache the output gradients.** Backpropagate the task loss through the full-precision network once, and store $\partial L / \partial z_k$ at the block's output for every calibration sample.

5. **Initialize the rounding variables.** Create a rounding variable $V$ for **every weight in the block**, across all of its layers, initialized so that the soft rounding reproduces the full-precision weights.

6. **Initialize the regularization schedule.** As in AdaRound, $\beta$ is annealed linearly during optimization (the official code first runs the initial 20% of iterations with no rounding regularizer, then anneals $\beta$ from 20 to 2 over the rest, so $t$ below counts iterations after that warmup):

    $$
    \beta_t
    =
    \beta_{\text{start}}
    +
    \frac{t}{T}
    \left(
    \beta_{\text{end}}
    -
    \beta_{\text{start}}
    \right).
    $$

<div class="paper-note">
    <strong>Note.</strong>
    <p>
        The paper describes the cached gradient as the gradient of the task loss at the block's output. In the official code, softmax is added at the end of the network and KL divergence is computed between the softmax outputs of the full-precision network and of the network quantized up to and including the current block. The error is backpropagated, and gradient for the current block is cached.
    </p>
</div>

<div class="algo-box" markdown="1">
<div class="algo-title">
<span>Algorithm</span>
<span>BRECQ</span>
</div>
<div class="algo-body" markdown="1">
 For each block, in order, and for each optimization iteration:

   1. Sample a mini-batch of calibration data.
   2. Compute the soft rounding values using the current $V$ (all layers in the block).
   3. Construct the quantized weights $\hat{W}$ for every layer in the block.
   4. Run the **whole block** on $\hat{x}$ with $\hat{W}$ to get the quantized block output $\hat{z}$.
   5. Compute the Fisher-weighted reconstruction loss:

      $$
      L(V)
      =
      \sum_k
      \left(\frac{\partial L}{\partial z_k}\right)^2
      \left(z_k - \hat{z}_k\right)^2
      +
      \lambda
      f_{\mathrm{reg}}(V).
      $$

   6. Backpropagate the loss with respect to $V$.
   7. Update $V$ using Adam.
   8. Update $\beta$ according to the annealing schedule.
   9. Go to step 1.

Once the iterations for the block are done:

- **Finalize the block**: convert the soft rounding values into hard binary decisions (round down or round up), construct the final quantized weights, and freeze the block.
- **Proceed to the next block**: repeat until every block (and every leftover layer) has been quantized.
</div>
</div>

## Conclusion

Thanks for reading! I hope this post showed you where layer-wise reconstruction falls short, and why BRECQ's move to block-level reconstruction is a good middle ground. In the next few posts, we'll look at modern methods that build on block-wise reconstruction for transformer models.

Here's a quick summary of BRECQ's design choices:

| Bit-width | Granularity | What's quantized |
|---|---|---|
| 2/3/4-bit weights; 4-bit activations (optional); first and last layer 8-bit | Per-channel weights, per-tensor activations| Weights and, optionally, activations (learned step size) |

## References

- Li et al., [BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction](https://arxiv.org/abs/2102.05426), ICLR 2021.
- Nagel et al., [Up or Down? Adaptive Rounding for Post-Training Quantization](https://arxiv.org/abs/2004.10568), ICML 2020.