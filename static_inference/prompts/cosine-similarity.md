## Mathematical Definition of Cosine Similarity

DreamZero jointly predicts action and future-video flow fields. Define the
ground-truth flow targets as

$$
u_A = \epsilon_A - A^*,
\qquad
u_V = \epsilon_V - Z^*,
$$

where $A^*$ is the ground-truth action chunk, $Z^*$ is the ground-truth video
latent, and $\epsilon_A,\epsilon_V$ are the fixed Gaussian noises used to
initialize the corresponding inference trajectories. Let $\hat u_{A,k}$ and
$\hat u_{V,k}$ be DreamZero's action and video flow predictions at inference
step $k$.

Calculate and save action and video cosine similarities separately at every
inference step.

### Action cosine similarity

DreamZero pads actions to `max_action_dim` and masks invalid or padded action
dimensions in its training loss. Apply the same `action_mask`, denoted by
$M_A$, before calculating the action cosine:

$$
\operatorname{cosine}_{A,k}
=
\frac{
\left\langle M_A \odot \hat u_{A,k},\; M_A \odot u_A \right\rangle
}{
\left\|M_A \odot \hat u_{A,k}\right\|_2
\left\|M_A \odot u_A\right\|_2
+ \varepsilon
}.
$$

Examples for which `has_real_action` is false do not have a defined action
cosine and must not contribute to the aggregate action-cosine statistic.

### Video cosine similarity

Calculate the video cosine directly between the predicted and ground-truth
video flow fields:

$$
\operatorname{cosine}_{V,k}
=
\frac{
\left\langle \hat u_{V,k},\; u_V \right\rangle
}{
\left\|\hat u_{V,k}\right\|_2
\left\|u_V\right\|_2
+ \varepsilon
}.
$$

If the predicted video flow has a smaller spatial shape than its target, crop
$u_V$ to the predicted height and width before calculating the cosine, matching
DreamZero's original dynamics-loss shape handling.

## DreamZero implementation notes

Use only the final transformer output: $\hat u_{A,k}$ is the action flow passed
to the action scheduler, and $\hat u_{V,k}$ is the video flow passed to the
video scheduler. Calculate both cosines before applying the scheduler update at
inference step $k$.
