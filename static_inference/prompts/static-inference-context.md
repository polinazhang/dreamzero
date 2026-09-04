## Context
**Static inference** is an analysis-only mode. Given demonstration samples, the VLA takes the state and other inputs, runs a “fake” forward denoising process, and does not execute the resulting actions. The predicted action and video flow fields are compared against their corresponding ground-truth flow targets.

**Static inference metrics** measure the discrepancy between the direction predicted by the model and the ground-truth direction it is expected to follow. One example of such metric is cosine similarity defined in `cosine-similarity.md`.

During static inference, the model should start from noise, and runs num_steps matching the model inference settings.