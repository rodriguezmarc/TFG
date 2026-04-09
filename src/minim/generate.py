"""
Definition:
Brief map of image-generation helpers for locally fine-tuned MINIM-compatible pipelines.
---
Results:
Loads fine-tuned pipelines and generates images from prompts or manifests.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


def _device_and_dtype(device: str | None) -> tuple[str, torch.dtype]:
    """
    ########################################
    Definition:
    Resolve the execution device and tensor dtype for inference.
    ---
    Params:
    device: Optional explicit device string.
    ---
    Results:
    Returns the device name and dtype pair used to load the pipeline.
    ########################################
    """
    if device is not None:
        return device, torch.float16 if device.startswith("cuda") else torch.float32
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def load_pipeline(model_path: Path | str, device: str | None = None) -> StableDiffusionPipeline:
    """
    ########################################
    Definition:
    Load a fine-tuned diffusers pipeline for image generation.
    ---
    Params:
    model_path: Model identifier or local model directory.
    device: Optional explicit device string.
    ---
    Results:
    Returns a ready-to-run `StableDiffusionPipeline`.
    ########################################
    """
    resolved_device, dtype = _device_and_dtype(device)
    pipeline = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype, safety_checker=None)
    pipeline = pipeline.to(resolved_device)
    pipeline.set_progress_bar_config(disable=False)
    return pipeline


def generate_images(
    model_path: Path | str,
    prompts: list[str],
    output_dir: Path,
    modality: str | None = None,
    device: str | None = None,
    seed: int | None = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
) -> list[Path]:
    """
    ########################################
    Definition:
    Generate images from an explicit prompt list.
    ---
    Params:
    model_path: Model identifier or local model directory.
    prompts: Text prompts to render.
    output_dir: Directory where generated images are written.
    modality: Optional modality prefix to prepend to prompts.
    device: Optional explicit device string.
    seed: Optional random seed for reproducibility.
    num_inference_steps: Number of denoising steps to execute.
    guidance_scale: Classifier-free guidance scale.
    ---
    Results:
    Returns the output paths of the generated images.
    ########################################
    """
    pipeline = load_pipeline(model_path, device=device)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    output_paths: list[Path] = []
    for index, prompt in enumerate(prompts):
        full_prompt = f"{modality}: {prompt}" if modality is not None and modality.strip() else prompt
        image = pipeline(
            full_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        output_path = output_dir / f"{index:05d}.png"
        image.save(output_path)
        output_paths.append(output_path)
    return output_paths


def read_prompts_from_manifest(manifest_path: Path) -> tuple[list[str], list[str]]:
    """
    ########################################
    Definition:
    Read prompts and modality tags from a manifest CSV.
    ---
    Params:
    manifest_path: Manifest path containing `text` and `modality` columns.
    ---
    Results:
    Returns the prompt list and aligned modality list.
    ########################################
    """
    prompts: list[str] = []
    modalities: list[str] = []
    with Path.open(manifest_path, encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            prompts.append(row["text"])
            modalities.append(row.get("modality", ""))
    return prompts, modalities


def generate_from_manifest(
    model_path: Path | str,
    manifest_path: Path,
    output_dir: Path,
    device: str | None = None,
    seed: int | None = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
) -> list[Path]:
    """
    ########################################
    Definition:
    Generate images for every prompt row in a manifest CSV.
    ---
    Params:
    model_path: Model identifier or local model directory.
    manifest_path: Manifest path containing prompts to evaluate.
    output_dir: Directory where generated images are written.
    device: Optional explicit device string.
    seed: Optional random seed for reproducibility.
    num_inference_steps: Number of denoising steps to execute.
    guidance_scale: Classifier-free guidance scale.
    ---
    Results:
    Returns the output paths of the generated images.
    ########################################
    """
    prompts, modalities = read_prompts_from_manifest(manifest_path)
    pipeline = load_pipeline(model_path, device=device)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    output_paths: list[Path] = []
    for index, (prompt, modality) in enumerate(zip(prompts, modalities)):
        full_prompt = f"{modality}: {prompt}" if modality.strip() else prompt
        image = pipeline(
            full_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        output_path = output_dir / f"{index:05d}.png"
        image.save(output_path)
        output_paths.append(output_path)
    return output_paths
