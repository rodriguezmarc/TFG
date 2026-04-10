# MINIM Integration Review Checklist

This list captures the remaining points that still need confirmation before the first production fine-tuning run.

## Licensing

- Confirm whether `https://github.com/WithStomach/MINIM` has an explicit software license. As of April 8, 2026, I could verify the public README and training instructions, but not a clear license grant.
- Do not vendor or copy MINIM source files into this repository until that license question is resolved.
- If the public repo remains unlicensed, keep the integration boundary external: local checkout path plus wrapper scripts only.

## MINIM runtime

- Verify against the current public MINIM repository that `train.sh` still consumes `MODEL_NAME` and `DATASET_NAME` and does not require additional mandatory edits.
- Decide whether the thesis implementation will permit random initialization. If yes, the local MINIM training code must be adapted to instantiate all core modules from config without relying on `MODEL_NAME`.
- Confirm online whether MINIM still expects the training CSV columns exactly as `path`, `Text`, and `modality`, including the capitalized `Text` field.
- Check whether MINIM natively consumes validation and test manifests or only a single training CSV.
- Confirm whether MINIM accepts absolute PNG paths without path rewriting inside its dataloader.
- Validate the pretrained checkpoint choice, image resolution, batch size, CUDA version, and Accelerate setup before the first real run.

## UKBB driver

- Confirm the actual UKBB directory layout before execution. The current driver supports several common filename patterns, but the real export layout must be checked on the dataset copy that will be used.
- Validate the metadata sidecar format used in UKBB. `sex`, `age`, `height`, `weight`, `pathology`, `ed_frame`, and `es_frame` are not all guaranteed to exist with the keys currently probed by the loader.
- Check whether UKBB labels are already in the canonical schema or must be remapped.
- Review ED/ES frame selection on real UKBB samples. The current fallback derives frames from labeled-frame volumes and needs confirmation on representative cases.
- Confirm whether disease labels used in prompts are clinically correct for the chosen UKBB subset, or whether they should be removed until a trusted metadata field is available.

## Data and evaluation

- Review whether `Cardiac MRI` is the final modality string expected by MINIM training or whether it should be standardized to another value.
- Confirm that the generated prompts are the final prompt format intended for fine-tuning and not only for preliminary export experiments.
- Decide whether ACDC and UKBB should be trained separately or merged into one combined run after both drivers are validated.
- Add a post-training evaluation plan: checkpoint selection, sample generation review, and quantitative metrics.
