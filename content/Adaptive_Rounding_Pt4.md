title: Part 4: Adaptive Rounding: LDLQ
date: 2026-11-20
category: Quantization

In my [last post](https://hello-fri-end.github.io/2026/10/part-3-adaptive-rounding-gptq/), I covered GPTQ. GPTQ quantizes the columns of a layer in a simple left-to-right order and, after rounding each one, adjusts the remaining columns to absorb the rounding error incurred as much as possible. Lazy batch updates and the Cholesky reformulation then made this fast enough for models with billions of parameters.

In this post, we look at LDLQ, introduced in the QuIP paper. LDLQ asks the same question the other way around and ends up making the **exact same rounding decisions as GPTQ**. So why bother? Because the new framing is cheaper to compute, comes with a proof of optimality that GPTQ inherits, and shows exactly where the approach breaks down. That last part motivates "incoherence processing", which QuIP introduced: rotating the weights and the Hessian with random orthogonal matrices before quantizing. Many quantization methods today build on the same idea, using rotations such as Hadamard transforms to reduce outliers.

## LDLQ: Pull Instead of Push

### Insight 1: Flip the Question

GPTQ (and OBQ before it) asks the following question:

> After quantizing a weight, how should we adjust the remaining unquantized weights to compensate for the resulting rounding error?

LDLQ asks:

> While quantizing weight $k$, how should we modify it to absorb the errors introduced by the previous $k-1$ quantized weights?

In other words, GPTQ pushes each error forward as soon as it happens. LDLQ pulls the accumulated errors in right before it rounds each weight. Let's see what that looks like on three weights.

### A Three-Weight Example

Consider three weights $\mathbf{w} = [w_1, w_2, w_3]$. Once $w_1$ is rounded, $w_2$ and $w_3$ haven't been rounded yet, so they are still free to move. Let $e_1 = \hat{w}_1 - w_1$ be the error from rounding $w_1$, and let $\delta_2, \delta_3$ be the shifts we'd like to give $w_2, w_3$. Recall the proxy loss we derived in the [Adaround blogpost](https://hello-fri-end.github.io/2026/08/part-1-adaptive-rounding-adaround/): $L = \mathbf{e}\mathbf{H}\mathbf{e}^\top$ with $\mathbf{e} = (e_1, \delta_2, \delta_3)$. Setting its derivatives with respect to $\delta_2$ and $\delta_3$ to zero gives a linear system. I derived it step by step in the [OBQ: Motivation](https://hello-fri-end.github.io/2026/09/part-2-adaptive-rounding-optimal-brain-quantizer-obq/#obq-motivation) section of my OBQ post, so here we start from its matrix form:

$$
\begin{bmatrix} H_{22} & H_{23} \\ H_{23} & H_{33} \end{bmatrix}
\begin{bmatrix} \delta_2 \\ \delta_3 \end{bmatrix}
= -e_1
\begin{bmatrix} H_{12} \\ H_{13} \end{bmatrix}
$$

We only apply $\delta_2$ now, so we eliminate $\delta_3$ from the first row by subtracting $H_{23}/H_{33}$ times the second row. That ratio is a Gaussian elimination multiplier, and the first row becomes:

$$
\left(H_{22} - \frac{H_{23}}{H_{33}} H_{23}\right)\delta_2
= -e_1\left(H_{12} - \frac{H_{23}}{H_{33}} H_{13}\right)
$$

Solving for $\delta_2$ gives:

$$
\delta_2 = -e_1\,\frac{H_{12} - H_{13}H_{23}/H_{33}}{H_{22} - H_{23}^2/H_{33}}
$$

If we had looked at $w_1$ and $w_2$ alone, we'd have gotten $-e_1 H_{12}/H_{22}$. The extra terms are $w_3$ taking its share.

Now round $w_2$. Its total error against the original weight is $e_2$ (its rounding error plus the shift it received). Only $w_3$ is left, so there is a single equation, $H_{33}\delta_3 = -(H_{13}e_1 + H_{23}e_2)$, and

$$
\delta_3 = -\frac{H_{13}\,e_1 + H_{23}\,e_2}{H_{33}}
$$

### From Three Weights to $n$

Compare the two results:

- **The last weight** uses plain ratios of the original $\mathbf{H}$: $H_{13}/H_{33}$ and $H_{23}/H_{33}$.
- **The weight before it** uses the same kind of ratio, but computed from an updated matrix. $H_{12}$ became $H_{12} - H_{13}H_{23}/H_{33}$, and $H_{22}$ became $H_{22} - H_{23}^2/H_{33}$. Those updates are what you get by eliminating $w_3$ from the equations.

This is Gaussian elimination, starting from the *last* variable. For $n$ weights, eliminate weight $n$, update the matrix, eliminate weight $n-1$ from the updated matrix, and so on. The multipliers at each stage are the coefficients. So, in general:

$$
w_k' = w_k - \sum_{j=1}^{k-1} m_{jk}\, e_j
$$

where $m_{jk}$ is the multiplier for weight $k$ from that elimination. Redoing the elimination at every step would be wasteful, so the next question is how to do it once.

### Insight 2: Read the Coefficients Off a Factorization

Gaussian elimination, with its bookkeeping written down, has a name: the LDL decomposition. We store all the multipliers in a matrix $\mathbf{U}$ and the leftover pivots in a diagonal matrix $\mathbf{D}$:

$$
\mathbf{H} = \mathbf{U}\mathbf{D}\mathbf{U}^\top
$$

where $\mathbf{U}$ is unit upper triangular. LDLQ does this elimination once, up front, and reads off one column per step:

$$
w_k' = w_k - \sum_{j=1}^{k-1} U_{jk}\, e_j
$$

For our three weights, $U_{23} = H_{23}/H_{33}$, $U_{13} = H_{13}/H_{33}$, and $U_{12}$ is the updated ratio from above.

Compare this with GPTQ. GPTQ factors $\mathbf{H}^{-1}$ once and reads a *row* per step to push an error forward. LDLQ factors $\mathbf{H}$ once and reads a *column* per step to pull errors in.

LDLQ is also cheaper to compute than OPTQ (another name for GPTQ): “OPTQ's implementation requires a matrix inversion of H, and two Cholesky decompositions.” LDLQ needs a single factorization of $\mathbf{H}$ and no inversion.

> **Note.** [My GPTQ post](https://hello-fri-end.github.io/2026/10/part-3-adaptive-rounding-gptq/) showed only one Cholesky factorization, the one of $\mathbf{H}^{-1}$. [The code](https://github.com/IST-DASLab/gptq/blob/main/gptq.py#L101-L103) gets $\mathbf{H}^{-1}$ with another one: it factors $\mathbf{H}$, uses that factor to invert $\mathbf{H}$, and then factors $\mathbf{H}^{-1}$. Inverting through the Cholesky factor gives the same $\mathbf{H}^{-1}$ as a direct inverse (up to floating-point error), and it is cheaper because $\mathbf{H}$ is symmetric positive definite.

### Insight 3: Same Rounding as GPTQ, Plus a Guarantee

LDLQ is not a better rounding scheme than GPTQ. Pushing an error forward early or pulling it in late gives the same value at the moment a weight is rounded, because every correction is linear in the past errors. The QuIP paper proves theoretically that the two produce the same rounding assignment, and the repo even includes [a script](https://github.com/Cornell-RelaxML/QuIP/blob/main/optq_ldlq_equiv.py) to reproduce it: on random data, the two implementations return identical outputs. The equivalence assumes the same $\mathbf{H}$ and the same quantization order.

In fact, among all methods that round one weight at a time using a linear correction from earlier errors (with coefficients fixed by $\mathbf{H}$, not by the weights), QuIP proves that LDLQ has the lowest layer error. This holds in the worst case and on average, when rounding to integers.

> **Note.** Two caveats from the paper. The guarantee covers the layer-wise proxy loss, not end-to-end accuracy. It also assumes rounding to all integers. With clamping to a real $b$-bit grid, the authors build a counterexample where clamped LDLQ does worse than nearest rounding, though they say it doesn't show up in practice.

With all three insights in place, the full LDLQ algorithm is:

<div class="algo-box" markdown="1">

<div class="algo-title">
<span>Algorithm</span>
<span>LDLQ</span>
</div>

<div class="algo-body" markdown="1">

1. Iterate through the network one layer at a time.
2. Run the calibration dataset through the (already quantized) earlier layers and collect input activations $\mathbf{X}$ for the current layer.
3. Compute the proxy Hessian with a dampening factor $\lambda$:
    $$
    \mathbf{H}
    =
    \mathbf{X}\mathbf{X}^{\top}
    +
    \lambda\mathbf{I}.
    $$
4. Factor $\mathbf{H}$:
    $$
    \mathbf{H}
    =
    \mathbf{U}\mathbf{D}\mathbf{U}^{\top}.
    $$
5. Express the weights in grid units. Initialize an error matrix $\mathbf{E} \in \mathbb{R}^{d_{\mathrm{row}} \times d_{\mathrm{col}}}$ with zeros. Column $j$ will hold $\mathbf{e}_j$.
6. For each block of $B$ columns:
    1. Collect the errors of all the columns rounded in earlier blocks, once, so they are not recomputed at every column.
    2. For each column $k$ in the block:
        1. Compute the correction from past errors, both those collected above and those of earlier columns inside this block:
            $$
            \mathbf{c}_k = -\sum_{j} U_{jk}\,\mathbf{e}_j.
            $$
        2. Shift the weight:
            $$
            \mathbf{w}_k' = \mathbf{w}_k + \mathbf{c}_k.
            $$
        3. Round and clamp:
            $$
            \hat{\mathbf{w}}_k = \mathrm{clamp}\left(\mathrm{round}(\mathbf{w}_k'),\ 0,\ 2^b - 1\right).
            $$
        4. Store $\mathbf{e}_k = \hat{\mathbf{w}}_k - \mathbf{w}_k$ in $\mathbf{E}$.
7. Move to the next layer and repeat.

</div>
</div>

## Where LDLQ Runs Out: Motivating Incoherence Processing

LDLQ works because later weights absorb the rounding errors of earlier ones. So when would it fail?

When $\mathbf{H}$ is mostly diagonal. Each weight's input is then unrelated to the others, so there is nobody to hand the error to, and LDLQ ends up rounding every weight on its own, like plain rounding.

**What would the best case look like?**

The opposite: every weight is entangled with many others, so a rounding error always lands somewhere that can soak it up. In terms of $\mathbf{H}$, its important directions are spread across many weights. This is called *incoherence*.

We can't choose $\mathbf{H}$, but we can change coordinates. Before quantizing, mix the weights with a random rotation (a random orthogonal matrix) so any layer looks like the best case. The output is unchanged because the rotation can be undone. This is incoherence processing, introduced by QuIP. Many follow-up papers improved upon the idea, and these days it's standard practice to apply some sort of rotation (e.g., with a Hadamard matrix) to weights and activations before quantizing them. We look into these methods in the next series.

## Conclusion

Thank you for reading! I hope this post showed that GPTQ and LDLQ are two framings of the same algorithm, one pushing errors forward and one pulling them in, and why the second framing is useful: it's cheaper, it gives a proof of optimality, and it shows exactly where the method fails. 

There is one more limitation of GPTQ/LDLQ: they quantize each layer independently, completely ignoring inter-layer dependencies.In reality, each layer receives quantized activations from the previous layer. Therefore, the actual reconstruction problem we should be solving is:

$$
\min_{\hat{W}} \| \hat{X}W - \hat{X}\hat{W} \|_F^2
$$

where $\hat{X}$ is the quantized input from the previous layer and $\hat{W}$ represents the quantized weights.

In the next post, we'll look at **Qronos**, which builds upon GPTQ by explicitly accounting for these inter-layer dependencies. See you there!

Here's a quick summary of LDLQ's design choices:

| Bit-width | What's quantized | What's not quantized | Compute precision | Scheme |
| --- | --- | --- | --- | --- |
| 2–4 bit (evaluated in the paper) | Weights only | Activations | FP16 (reference implementation) | Asymmetric, unsigned grid, per-row scale and zero-point (baseline); symmetric, unsigned grid, per-tensor scale (with incoherence processing) |

## References and Further Reading

1. [QuIP: 2-Bit Quantization of Large Language Models With Guarantees](https://arxiv.org/abs/2307.13304)
2. [Cornell-RelaxML/QuIP](https://github.com/Cornell-RelaxML/QuIP)
3. [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)