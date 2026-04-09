"""
Definition:
Brief map of evaluation helpers for generated MINIM images.
---
Results:
Computes image-quality metrics and writes evaluation reports.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from scipy.linalg import sqrtm

try:
    from pytorch_msssim import ms_ssim
except ImportError:  # pragma: no cover - dependency is optional until runtime evaluation
    ms_ssim = None


def _load_rgb_images(paths: list[Path], image_size: int = 299) -> torch.Tensor:
    """
    ########################################
    Definition:
    Load and resize a batch of RGB images for metric computation.
    ---
    Params:
    paths: Image paths to load.
    image_size: Target square image size.
    ---
    Results:
    Returns a stacked tensor of resized images.
    ########################################
    """
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    images = [transform(Image.open(path).convert("RGB")) for path in paths]
    return torch.stack(images)


def _real_paths_from_manifest(manifest_path: Path) -> list[Path]:
    """
    ########################################
    Definition:
    Read the real-image paths referenced by a manifest CSV.
    ---
    Params:
    manifest_path: Manifest path containing the `path` column.
    ---
    Results:
    Returns the ordered list of real image paths.
    ########################################
    """
    paths: list[Path] = []
    with Path.open(manifest_path, encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            paths.append(Path(row["path"]))
    return paths


def _generated_paths_from_dir(generated_dir: Path) -> list[Path]:
    """
    ########################################
    Definition:
    Discover generated image files inside one output directory.
    ---
    Params:
    generated_dir: Directory containing generated images.
    ---
    Results:
    Returns the sorted list of generated image paths.
    ########################################
    """
    return sorted(path for path in generated_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})


def load_inception_model(device: str) -> torch.nn.Module:
    """
    ########################################
    Definition:
    Load the Inception v3 feature extractor used by evaluation metrics.
    ---
    Params:
    device: Execution device for the model.
    ---
    Results:
    Returns the initialized feature extractor in evaluation mode.
    ########################################
    """
    weights = models.Inception_V3_Weights.DEFAULT
    inception = models.inception_v3(weights=weights, transform_input=False)
    inception.fc = torch.nn.Identity()
    inception.eval()
    inception.to(device)
    return inception


def _extract_features(model: torch.nn.Module, images: torch.Tensor, device: str) -> np.ndarray:
    """
    ########################################
    Definition:
    Extract feature vectors for a batch of images.
    ---
    Params:
    model: Feature extractor model.
    images: Batch of images to encode.
    device: Execution device for the model.
    ---
    Results:
    Returns the extracted features as a NumPy array.
    ########################################
    """
    with torch.no_grad():
        features = model(images.to(device))
    return features.detach().cpu().numpy()


def calculate_fid(real_features: np.ndarray, generated_features: np.ndarray) -> float:
    """
    ########################################
    Definition:
    Compute the Fréchet Inception Distance between two feature sets.
    ---
    Params:
    real_features: Feature vectors from real images.
    generated_features: Feature vectors from generated images.
    ---
    Results:
    Returns the FID score as a float.
    ########################################
    """
    mu_real = np.mean(real_features, axis=0)
    mu_generated = np.mean(generated_features, axis=0)
    cov_real = np.cov(real_features, rowvar=False)
    cov_generated = np.cov(generated_features, rowvar=False)
    diff = mu_real - mu_generated
    covmean = sqrtm(cov_real @ cov_generated)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov_real + cov_generated - 2 * covmean))


def calculate_inception_score(logits: torch.Tensor) -> float:
    """
    ########################################
    Definition:
    Compute the Inception Score from model logits.
    ---
    Params:
    logits: Inception logits for generated images.
    ---
    Results:
    Returns the Inception Score as a float.
    ########################################
    """
    probs = F.softmax(logits, dim=1).cpu().numpy()
    marginal = np.mean(probs, axis=0, keepdims=True)
    kl = probs * (np.log(probs + 1e-12) - np.log(marginal + 1e-12))
    return float(np.exp(np.mean(np.sum(kl, axis=1))))


def calculate_ms_ssim(real_images: torch.Tensor, generated_images: torch.Tensor) -> float:
    """
    ########################################
    Definition:
    Compute the Multi-scale Structural Similarity score for paired images.
    ---
    Params:
    real_images: Real-image tensor batch.
    generated_images: Generated-image tensor batch.
    ---
    Results:
    Returns the MS-SSIM score as a float.
    ########################################
    """
    if ms_ssim is None:
        raise ImportError("`pytorch-msssim` is required to compute MS-SSIM.")
    return float(ms_ssim(real_images, generated_images, data_range=1.0, size_average=True).item())


def evaluate_from_manifest(
    real_manifest_path: Path,
    generated_dir: Path,
    device: str = "cpu",
    output_json_path: Path | None = None,
) -> dict[str, float | int]:
    """
    ########################################
    Definition:
    Evaluate generated images against the real images referenced in a manifest.
    ---
    Params:
    real_manifest_path: Manifest path for the real evaluation set.
    generated_dir: Directory containing generated images.
    device: Execution device for metric computation.
    output_json_path: Optional path where metrics are serialized as JSON.
    ---
    Results:
    Returns the computed evaluation metrics.
    ########################################
    """
    real_paths = _real_paths_from_manifest(real_manifest_path)
    generated_paths = _generated_paths_from_dir(generated_dir)
    sample_count = min(len(real_paths), len(generated_paths))
    if sample_count == 0:
        raise ValueError("No images available for evaluation.")

    real_images = _load_rgb_images(real_paths[:sample_count])
    generated_images = _load_rgb_images(generated_paths[:sample_count])
    inception = load_inception_model(device)

    real_features = _extract_features(inception, real_images, device)
    generated_features = _extract_features(inception, generated_images, device)
    with torch.no_grad():
        generated_logits = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT).to(device).eval()(
            generated_images.to(device)
        )

    metrics = {
        "sample_count": sample_count,
        "fid": calculate_fid(real_features, generated_features),
        "is": calculate_inception_score(generated_logits),
        "ms_ssim": calculate_ms_ssim(real_images, generated_images),
    }
    if output_json_path is not None:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    """
    ########################################
    Definition:
    Parse CLI arguments for standalone metric evaluation.
    ---
    Params:
    None.
    ---
    Results:
    Returns the parsed CLI namespace.
    ########################################
    """
    parser = argparse.ArgumentParser(description="Evaluate generated MINIM images against a real-image manifest")
    parser.add_argument("--real-manifest", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """
    ########################################
    Definition:
    Execute standalone metric evaluation from CLI arguments.
    ---
    Params:
    None.
    ---
    Results:
    Prints the computed metrics as JSON.
    ########################################
    """
    args = parse_args()
    metrics = evaluate_from_manifest(
        real_manifest_path=args.real_manifest,
        generated_dir=args.generated_dir,
        device=args.device,
        output_json_path=args.output_json,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
