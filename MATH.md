# Mathematics

## Explained Simply: The Essence of Hyper-Merging

The Hyper-Merging Algorithm focuses on understanding the nuances in the weights of Stable-Diffusion neural networks. By comparing different models to an average or base model, the algorithm identifies new "directions" that represent how each model varies. This process is iterative, allowing us to discover multiple such directions, thereby capturing finer differences between models. This capability provides a valuable toolkit for enhancing neural network models or exploring new creative avenues.

## Mathematical Framework: Digging Deeper into the Equations

Consider a neural network weights, specifically a Stable-Diffusion one, denoted as $W_{m,l,\theta_l}$, where:

- $m$ represents a particular model,
- $l$ represents a layer within the network, and
- $\theta_l$ represents the parameters specific to that layer.

Now, let's introduce a new concept: $\tilde{W}_{l,\theta_l}$, which can be thought of as an aggregate representation. This could be the average of $M$ models or any other chosen model, perhaps the base model. We can then create the differential weight $\Delta W_{m,l,\theta_l} = W_{m,l,\theta_l} - \tilde{W}_{l,\theta_l}$.

Essentially, this differential weight signifies the direction from which the weights of a specific model deviate from the aggregate or base model. When multiple models are in play, each one points in a distinct direction.

Our objective is to find a novel direction, denoted as $\delta W_{l,\theta_l}$. When this direction is scaled by a factor $\lambda_m$, it should approximate the original direction $\Delta W_{m,l,\theta_l}$. This can be achieved by minimizing the following loss function:

$$
\sum _{m=1}^M \sum _{l=1}^L \sum _{\theta _l=1}^{\Theta _l} \left(\Delta W_{m,l,\theta _l}-\lambda _m \delta W_{l,\theta _l}\right)^2
$$

To minimize this loss, we can employ the following update equations, which solve for each variable iteratively:

$$
\lambda _m\to \frac{\sum _{\theta _l=1}^{\Theta _l} \sum _{l=1}^L \delta W_{l,\theta _l} \Delta W_{m,l,\theta _l}}{\sum _{\theta _l=1}^{\Theta _l} \sum _{l=1}^L \delta W_{l,\theta _l}^2}
$$

$$
\delta W_{l,\theta _l}\to \frac{\sum _{m=1}^M \lambda _m \Delta W_{m,l,\theta _l}}{\sum _{m=1}^M \lambda _m^2}
$$

But why stop at just one new direction? By modifying $\Delta W_{m,l,\theta _l} \to \Delta W_{m,l,\theta _l}-\lambda _m \delta W_{l,\theta _l}$ and repeating the process, we can uncover additional directions — second, third, and so on.

Why is this valuable? Imagine you have both realistic and cartoon models. The first direction might capture the transition from realism to a more illustrative style. By discovering subsequent directions, we can introduce finer details and enhancements to the models, opening up new creative possibilities.
