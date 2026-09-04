## Symbols and WAM forward function

Let the world-action model be denoted by

$$
  (\hat u_A,\hat u_V)
=
  \Psi_\theta
  \left(
  A^{\tau_A},
  Z_{\mathrm{future}}^{\tau_V},
  C_{\mathrm{obs}},
  h_\ell,
  h_s,
  t_A,
  t_V
  \right),
$$

  where

$$
  C_{\mathrm{obs}} =
  h_{\mathrm{CLIP}}(I_{\mathrm{obs}}),
  Y_{\mathrm{VAE}}(I_{\mathrm{obs}}),
  KV_{\mathrm{history}}

$$


$Z^{\tau_V}$ is the input latent visual embedding, $\Psi_\theta$ is the joint video and action predictor transformer, $\theta$ are the model parameters, and $\hat u_A$ is the predicted clean action (or equivalent action-space output of the policy head).

  The roles are:

  - $A^{\tau_A}$: noisy action trajectory being denoised.
  - $Z_{\mathrm{future}}^{\tau_V}$: noisy future-video latent being denoised.
  - $I_{\mathrm{obs}}$: real observed camera image or frames.
  - $h_{\mathrm{CLIP}}(I_{\mathrm{obs}})$: semantic first-image conditioning.
  - $Y_{\mathrm{VAE}}(I_{\mathrm{obs}})$: spatial first-frame conditioning used by the Wan I2V path.
  - $KV_{\mathrm{history}}$: cached tokens from previously observed/generated frame chunks.
  - $t_A,t_V$: action and video diffusion timesteps. Their embeddings are $h_{t_A}$ and $h_{t_V}$, constructed inside the DiT.



The noisy latents are defined as


$$
A^{\tau_A}=(1-\tau_A)A^*+\tau_A\epsilon_A,
\qquad
Z^{\tau_V}=(1-\tau_V)Z^*+\tau_V\epsilon_V.
$$


where $\epsilon \sim \mathcal{N}(0,I)$, $A^*$ is the ground-truth action, and $Z^*$ is the ground truth video.


## Vision Grad Norm: local gradient norm on the target variable

As an example, we consider the vision embedding and compute the gradient of the loss with respect to $h_v$ to measure local sensitivity, where

$$
  h_v =
  h_{\mathrm{CLIP}}(I_{\mathrm{obs}}),
  Y_{\mathrm{VAE}}(I_{\mathrm{obs}}),
$$

(Basically $h_v$ is the part of $C_{\mathrm{obs}}$ excluding $KV_{\mathrm{history}}$)

$$
g_v (action) := \nabla_{h_v} L_{action}.
$$

$$
g_v (video) := \nabla_{h_v} L_{video}.
$$

Here, when you take the gradient, you should take the gradient jointly on $h_{\mathrm{CLIP}}(I_{\mathrm{obs}})$ and $Y_{\mathrm{VAE}}(I_{\mathrm{obs}})$. There should be no separate sensiticity score for CLIP and VAE.

The scalar sensitivity score is defined as the gradient norm

$$
S_v := \|g_v\|_2.
$$

In practice, for each evaluation example, one performs a forward pass to compute $L$, backpropagates through the model while treating $h_v$ as the differentiation target, and records $\| \nabla_{h_v} L \|_2$.




This quantity measures the first-order sensitivity of the loss to infinitesimal perturbations of the vision embedding at the current point.


### Equivalence of $\nabla_{\delta_v} L(0)$ and $\nabla_{h_v} L$

We note that $\nabla_{\delta_v} L(0) = \nabla_{h_v} L$ since $\delta_v$ is an additive reparameterization of $h_v$.


## Special notes for DreamZero implementation

During inference dreamzero will have multiple steps (keep it consistent with default number). The model should calculate and save grad norm of all those inference steps with pure noise instead of 1 step with randomly sampled time and noise deducted by ground truth action.

DreamZero uses $\text{noisy}=(1-t)\cdot\text{actions}+t\cdot\text{noise}$. Therefore, pure noise starts from (t=1).

During inference, evaluate at all 16 default DreamZero inference steps, using the exact timesteps produced by its inference scheduler. “The first diffusion step” means the initial (t=1) pure-noise step.

Instead of one randomly sampled training timestep with noise added to the ground-truth action, initialize the action and future- video latents from pure noise and calculate/save the vision gradient norm at every inference step.

For coding, create separate additive perturbations for DreamZero’s two visual embeddings:

delta_clip = torch.zeros_like(h_clip).requires_grad_(True)
delta_vae = torch.zeros_like(y_vae).requires_grad_(True)

Add them to h_clip and y_vae, keep the warmed visual KV cache fixed, and call torch.autograd.grad(loss, (delta_clip, delta_vae)).
Since each delta is an additive reparameterization, its gradient at zero equals the gradient with respect to the corresponding visual embedding.