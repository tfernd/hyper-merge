# Hyper-Merge: Advanced Weight Merging for Stable-Diffusion

### TLDR

Introducing **Hyper-Merge**: a new algorithm designed to merge multiple Stable-Diffusion models more effectively than basic linear combinations. By applying advanced mathematical techniques, Hyper-Merge creates a new set of model weights. It allows for smooth transitions between different models using one or two multipliers. This offers a more precise and efficient approach to building stable neural models. The resulting hyper-model can also be converted into a **LoRA** file.

---

### Introduction

Combining Stable-Diffusion models in the field of text-to-image neural networks has always been challenging due to the issue of incompatible internal weights. This makes the merging process uncertain and complex.

**Hyper-Merge** provides a mathematical solution to address these challenges. Unlike traditional methods relying on simple linear combinations, Hyper-Merge uses a specialized loss function to generate a new set of optimized model weights. The optimization process is efficient, eliminating the need for iterative gradient descent.

We offer two versions of Hyper-Merge: a 1D framework for straightforward applications and a 2D framework for those needing more control and complexity. Both are designed to produce hyper-models that are stable and versatile.

As for the application of the differential weights produced by Hyper-Merge, they can be converted into a **LoRA** file. Typically, generating a LoRA file requires a fine-tuned model and a base model, with the weight differences then processed through SVD decomposition to save space. Since Hyper-Merge creates a differential model, these weights can be directly converted into a LoRA file!

So why settle for anything less? Dive into the world of Hyper-Merge and experience the future of neural network model merging. 🌠

## Visual Guide to Hyper-Merge 🎨

Below is a visualization designed to demystify the underlying mechanics of Hyper-Merge. The diagram is a 2D simplification, where each point represents one of eight unique models. Their respective weights have been distilled down to fit this 2D projection. The pivotal red point symbolizes the collective centroid of these models in the weight space.

<img src="assets/Arrows.jpg" height=256 alt="Model arrows">

Take notice of the solid arrows—these depict the hyper-models synthesized through Hyper-Merge's advanced algorithm. Each hyper-model is fine-tuned via multipliers to approximate the individual models. In contrast, the dashed arrows serve to highlight the error vectors, revealing deviations between the actual models and their hyper-model approximations. **Disclaimer**: _the image is just random points to illustrate the mechanism._

Imagine extrapolating this "easy-to-grasp" 2D diagram into a sprawling, multi-gigabyte, hyper-dimensional matrix. Yes, you guessed it—that's Hyper-Merge flexing its computational muscles. 💪

## **Real-World Hyper-Merge Examples** 🌍

Feast your eyes on the hyper-merged models showcased below! These captivating examples feature a fusion of the following cutting-edge models:

1. Juggernaut Aftermath
2. Realistic Vision - v5.1
3. Reliberate - v2.0
4. epiCRealism - Natural Sin RC1 VAE
5. epiCRealism Pure Evolution - v5
6. CyberRealistic - v3.3
7. epiCDream - Lullaby
8. Photon - v1

The accompanying GIFs offer a dynamic view of this potent amalgamation in action. Each animated sequence illustrates different multipliers, unveiling a spectrum of composite models produced by Hyper-Merge. 🌈

<div>
   <img src="assets/animation-1.gif" height=320>
   <img src="assets/animation-2.gif" height=320>
</div>

---

## **TODO Checklist** 🗂️

- [ ] Generate a LoRA using identified differential weights.
- [ ] Implement code for 2-dimensional differential weight analysis.

---

## **Deconstructing Criticism: LoRA vs Hyper-Merge** 💡

You might wonder, "Why not simply extract a LoRA from the models?"

While a LoRA traditionally involves only two models—a fine-tuned model and its base—it calculates differential weights between these two before performing Singular Value Decomposition (SVD) to cut down the number of parameters. Sounds simple, right? 🤔

However, Hyper-Merge takes it up a notch. Though based on similar foundational concepts like differential weights and multipliers, Hyper-Merge demands more than just 2 models for its magic to happen. 🌟 Moreover, the differential weights in this approach are tailored to minimize the distance to multiple fine-tuned models. In essence, it achieves a very different and more expansive outcome compared to a traditional LoRA.

---

## **Getting Started: No Installation Needed!** 🚀

Eager to dive in? You're in luck; there's no need to install additional software! Activate the `venv` from your Automatic1111 and fire up the [merge.ipynb](merge.ipynb) notebook using Jupyter. Here are the step-by-step commands to get you started:

```bash
# Activate your virtual environment
source PATH-TO-WEBUI/venv/scripts/activate

# Clone the repository
git clone https://github.com/tfernd/hyper-merge

# Install the required packages (This won't affect your existing packages, promise! 👍)
pip install -r requirements.txt

# Open the Jupyter notebook
jupyter notebook ./merge.ipynb
```

(Note: Please make sure to replace `PATH-TO-WEBUI` with the actual path to your `venv` directory.)

---

## Mathematics

Let's dive deeper into the mathematical intricacies behind the Hyper-Merge algorithm.

### Notation

- $W_{m,l,\theta_l}$: Weights of the model $ m $ at layer $l$ with parameters $\theta_l$
- $W'_{l,\theta_l}$: New set of weights for layer $ l $ with parameters $ \theta_l $
- $\lambda_m$: Parameter multiplied by $W'_{l,\theta_l}$ to approximate $W_{m,l,\theta_l}$

### Approach 1: Basic Weight Merging

The first approach is to minimize the following loss function:

<!-- $$
\sum_{m=1}^M \sum_{l=1}^L \sum_{\theta_l=1}^{\Theta_l} \left( W_{m,l,\theta_l} - \lambda_m W'_{l,\theta_l} \right)^2
$$ -->

![](assets/eq/loss1.png)

To minimize this loss, we can use the following update equations:

<!-- $$
\lambda_m \to \frac{\sum_{\theta_l=1}^{\Theta_l} \sum_{l=1}^L W_{m,l,\theta_l} W'_{l,\theta_l}}{\sum_{\theta_l=1}^{\Theta_l} \sum_{l=1}^L {W'}_{l,\theta_l}^2}
$$

$$
W'_{l,\theta_l} \to \frac{\sum_{m=1}^M \lambda_m W_{m,l,\theta_l}}{\sum_{m=1}^M \lambda_m^2}
$$ -->

![](assets/eq/params1.png)

Note: This method does not guarantee a mathematical minimum but empirically results in a good approximation. The scales and weights are updated sequentially for a few iterations until local/global minima is achieved.

### Approach 1b: Differential Weight Merging

A more robust approach introduces the concept of differential weights, ${\delta W}_{l,\theta_l}$, and a base or average weight, $\tilde{W}_{l,\theta_l}$ taken from the base model or the average model. The loss function is then:

<!-- $$
\sum_{m=1}^M \sum_{l=1}^L \sum_{\theta_l=1}^{\Theta_l} \left( W_{m,l,\theta_l} - (\tilde{W}_{l,\theta_l} + \lambda_m {\delta W}_{l,\theta_l}) \right)^2
$$ -->

![](assets/eq/loss2.png)

The update equations for this approach are:

<!-- $$
\lambda_m \to \frac{\sum_{\theta_l=1}^{\Theta_l} \sum_{l=1}^L {\delta W}_{l,\theta_l} \left( W_{m,l,\theta_l} - \tilde{W}_{l,\theta_l} \right)}{\sum_{\theta_l=1}^{\Theta_l} \sum_{l=1}^L {\delta W}_{l,\theta_l}^2}
$$

$$
{\delta W}_{l,\theta_l} \to \frac{\sum_{m=1}^M \lambda_m \left( W_{m,l,\theta_l} - \tilde{W}_{l,\theta_l} \right)}{\sum_{m=1}^M \lambda_m^2}
$$ -->

![](assets/eq/params1b.png)

### Advantages of Approach 1b

This second approach offers several advantages:

- It allows for more flexible scaling of $\lambda$ and ${\delta W}$.
- $\lambda$ can be easily re-scaled to fit within a range of $[-1, 1]$, making it compatible with commonly used LoRA multiplier scales.
- User might select different $lambda$ after finding ${\delta W}$ to get new models.

### Approach 2: 2-Dimensional Scaling

Until now, we've considered weight scaling along a 1-dimensional line, akin to LoRA models. But what if we extend this into a 2-dimensional plane? One axis could distinguish between, say, realistic and digital art styles, while the other could capture finer variations.

The expanded loss function for this plane-based approach is:

<!-- $$
\sum_{m=1}^M \sum_{l=1}^L \sum_{\theta_l=1}^{\Theta_l} \left( W_{m,l,\theta_l} - \left( \tilde{W}_{l,\theta_l} + \lambda_m {\delta W}_{l,\theta_l} + \kappa_m {\delta Q}_{l,\theta_l}  \right) \right)^2
$$ -->

![](assets/eq/loss3.png)

Where $\delta W$ and $\delta Q$ are the two differential weights and $\lambda$ and $\kappa$ are the two multipliers.

<!-- $$
\lambda _m\to \frac{\sum _{\theta _l=1}^{\Theta _l} \sum _{l=1}^L {\delta W}_{l,\theta _l} \left(W_{m,l,\theta _l}-\tilde{W}_{l,\theta _l}-\kappa _m {\delta Q}_{l,\theta _l}\right)}{\sum _{\theta
   _l=1}^{\Theta _l} \sum _{l=1}^L {\delta W}_{l,\theta _l}^2}
$$

$$
\kappa _m\to \frac{\sum _{\theta _l=1}^{\Theta _l} \sum _{l=1}^L {\delta Q}_{l,\theta _l} \left(W_{m,l,\theta _l}-\tilde{W}_{l,\theta _l}-\lambda _m {\delta W}_{l,\theta _l}\right)}{\sum _{\theta
   _l=1}^{\Theta _l} \sum _{l=1}^L {\delta Q}_{l,\theta _l}^2}
$$

$$
{\delta W}_{l,\theta _l}\to \frac{\sum _{m=1}^M \lambda _m \left(W_{m,l,\theta _l}-\kappa _m {\delta Q}_{l,\theta _l}-\tilde{W}_{l,\theta _l}\right)}{\sum _{m=1}^M \lambda _m^2}
$$

$$
{\delta Q}_{l,\theta _l}\to \frac{\sum _{m=1}^M \kappa _m \left(W_{m,l,\theta _l}-\lambda _m {\delta W}_{l,\theta _l}-\tilde{W}_{l,\theta _l}\right)}{\sum _{m=1}^M \kappa _m^2}
$$ -->

![](assets/eq/params2.png)

## What is LoRA?

LoRA: Low-Rank Adaptation
Low-Rank Adaptation of Large Language Models (LoRA) is a training method that accelerates the training of large models while consuming less memory. It adds pairs of rank-decomposition weight matrices (called update matrices) to existing weights, and only trains those newly added weights. This has a couple of advantages:

Previous pretrained weights are kept frozen so the model is not as prone to catastrophic forgetting.
Rank-decomposition matrices have significantly fewer parameters than the original model, which means that trained LoRA weights are easily portable.
LoRA matrices are generally added to the attention layers of the original model. 🧨 Diffusers provides the load_attn_procs() method to load the LoRA weights into a model’s attention layers. You can control the extent to which the model is adapted toward new training images via a scale parameter.
The greater memory-efficiency allows you to run fine-tuning on consumer GPUs like the Tesla T4, RTX 3080 or even the RTX 2080 Ti! GPUs like the T4 are free and readily accessible in Kaggle or Google Colab notebooks.
