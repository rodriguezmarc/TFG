# MINIM Integration Review Checklist

This list captures the parts of the codebase that still need confirmation before the first production fine-tuning run.

## MINIM integration

- Verify against the current public MINIM repository that `train.sh` still consumes `MODEL_NAME` and `DATASET_NAME` and does not require additional mandatory environment variables or config edits.
- Confirm online whether MINIM still expects the training CSV columns exactly as `path`, `Text`, and `modality`, including the capitalized `Text` field.
- Check whether MINIM supports validation and test manifests natively or only a single training CSV. If it only supports one CSV, decide whether validation must be implemented outside MINIM.
- Confirm whether MINIM accepts absolute PNG paths without path rewriting inside its dataloader.
- Validate the pretrained checkpoint choice, image resolution, batch size, and CUDA/Accelerate requirements before the first real run.
- Review whether the public MINIM codebase is the correct target repository and branch for this thesis work, or whether another internal fork should be used.

## UKBB driver

- Confirm the actual UKBB directory layout before execution. The current driver supports several common filename patterns, but the real acquisition/export layout must be checked on the dataset copy that will be used.
- Validate the metadata sidecar format used in UKBB. `sex`, `age`, `height`, `weight`, `pathology`, `ed_frame`, and `es_frame` are not all guaranteed to exist with the keys currently probed by the loader.
- Check whether UKBB labels are already in the canonical schema or must be remapped. The current implementation relies on optional metadata hints for this.
- Review ED/ES frame selection on real UKBB samples. The current fallback derives frames from labeled-frame volumes and needs confirmation on representative cases.
- Confirm whether disease labels used in prompts are clinically correct for the chosen UKBB subset, or whether they should be removed until a trusted metadata field is available.

## Data and prompt pipeline

- Review whether `Cardiac MRI` is the final modality string expected by MINIM training or whether it should be standardized to another value.
- Confirm that the generated prompts are the final prompt format intended for fine-tuning and not only for preliminary export experiments.
- Decide whether ACDC and UKBB should be trained separately or merged into a combined manifest once both drivers are validated.
- Review whether overlay outputs under `output/images/<dataset>/masked/` should stay as diagnostics only or become part of another supervised task.
- Add a post-training evaluation plan: checkpoint selection, sample generation review, and quantitative metrics.

