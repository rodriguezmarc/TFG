Base model descriptors live here.

Real MINIM training expects a downloaded Stable Diffusion v1.4 checkpoint at:

- `models/base/stable-diffusion-v1-4/`

If that folder exists, it must contain the full diffusers snapshot.
An empty or partial folder is treated as an invalid installation and the pipeline will fail early.

If the folder does not exist, the pipeline falls back to the Hugging Face model id
`CompVis/stable-diffusion-v1-4`, which requires online access.

Pipeline validation can use the lightweight mock model descriptor at:

- `models/base/mock-minim/`

The mock model is selected with `--backend mock`. It does not contain diffusion weights; it preserves
the training and generation artifact contract so the rest of the pipeline, including evaluation and
best-checkpoint ranking, can run on low-resource machines.
