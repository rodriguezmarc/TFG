# Mock MINIM Base Model

This directory identifies the lightweight mock MINIM model backend.

The mock backend does not contain Stable Diffusion weights. It simulates the
model-side contract by writing checkpoint metadata and deterministic placeholder
generated images. The normal evaluation layer still consumes those generated
images and writes metrics.
