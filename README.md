# Hyper-LoRA

Hyper-LoRA changes the landscape of LoRA by introducing advanced functionalities, including the ultra-fast extraction of LoRA from pre-trained Stable-Diffusion (SD) models (under 3 seconds), seamless merging of a LoRA into a model, and the unique feature of Hyper-LoRA.

Users of SD often struggle with selecting an optimal base model from a plethora of models performing slightly different tasks. Furthermore, enthusiasts often attempt to blend models manually through trial and error, which is time-consuming and less efficient. Hyper-LoRA addresses this challenge by automating the process and optimizing model merging.

## Mathematical Framework

Hyper-LoRA employs an advanced mathematical framework to optimize model merging and weight distribution. The process involves minimizing a loss function over multiple models, providing a better means of consolidating various models into one optimal solution.

Let's break this down:

Consider $M$ to be the total number of models that need to be merged, and $N$ represents the total number of weights an SD model contains. Furthermore, let $\Theta_n$ be the number of weights associated with a certain layer $n$.

We define $dW$ as the difference between a fine-tuned model (such as one produced by Dreambooth) and a base model. However, this can be extended to an arbitrary set of models.

The loss function that we seek to minimize is as follows:

$$
\sum_m^M\sum_n^N\sum_{\theta_n}^{\Theta_n} \left(
 dW_{n, \theta_n}^m - dW'_{n, \theta_n}
\right)^2,
$$

In this equation, $dW'$ represents a newly found differential weight that minimizes the distance to all other models. It can be proven that this differential weight should be the mean of all weights, thus consolidating multiple models into a single one by averaging them corresponds to minimizing this loss.

To refine this process further, a smarter way to define a loss function would be:

$$
\sum_m^M\sum_n^N\sum_{\theta_n}^{\Theta_n} \left(
 dW_{n, \theta_n}^m - \lambda_m dW'_{n, \theta_n}
\right)^2,
$$

Here, $\lambda_m$ is a scaling factor applied to the differential weight for each model, functioning similarly to multipliers in LoRAs. This introduces an adaptive measure in the merging process, allowing some models to have higher multipliers, others lower, and even potentially negative ones.

The role of LoRAs becomes crucial here. Given that each model consumes a substantial amount of memory (2 GB each), merging multiple models would require significant computational resources. LoRAs enable us to extract the essential aspects of each model first, compute the differential weights per layer, and then move computations between the GPU and CPU as necessary.

Finally, the scale factor $\lambda$ and the differential $dW$ can be computed interactively by:

$$
dW'_{n, \theta_n} = \frac{\sum_m^M \lambda_m dW_{n, \theta_n}^m}{\sum_m^M \lambda_m^2}
$$

and

$$
\lambda_m = \frac{\sum_n^N\sum_{\theta_n}^{\Theta_n} dW_{n, \theta_n}^m  dW'_{n, \theta_n}}{\sum_n^N\sum_{\theta_n}^{\Theta_n}  dW'^{2}_{n, \theta_n}}
$$

In essence, this mathematical framework allows Hyper-LoRA to efficiently and effectively merge multiple models, optimizing weight distribution and reducing the computational resources required.

## Features

- **LoRA Extraction**: Quickly extract LoRA parameters from base and tuned models without model instantiation using torch.svd_lowrank for rapid SVD.
- **LoRA Merge**: Seamlessly reintegrate LoRA parameters into a model without extensive model instantiation.
- **Hyper-LoRA**: Advanced feature to optimize model merging. (Description to be added)
- **Hyper-LoRA Orthogonalization**: Determine primary and secondary directions for LoRA. (Work in Progress)

## Advantages

Hyper-LoRA offers a significant advantage over other implementations like [https://github.com/bmaltais/kohya_ss](https://github.com/bmaltais/kohya_ss) by eliminating the need for model instantiation. This leads to faster extraction, saving both time and computational resources.

## Future Work

- [ ] Integrate with AUTOMATIC1111
- [ ] Develop a standalone GUI using Gradio
- [ ] Introduce secondary direction to Hyper-LoRA
- [ ] Provide Google Colab Support for Civitai Models: To streamline the process further, we plan to add a Google Colab notebook feature. This will allow users to extract LoRA from Civitai models without the need for local downloads, thus saving space and enhancing ease of use.

## Contributing

Contributions to Hyper-LoRA are always welcome! If you discover a bug, have a feature request, or wish to contribute code, please open an issue or submit a pull request.

## License

Hyper-LoRA is licensed under the [MIT License](LICENSE).
