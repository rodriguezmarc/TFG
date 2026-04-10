# Proposed Fine-Tuning Architecture

## Goal

Build the fine-tuning layer on top of the current driver-based `src/data` architecture without duplicating preprocessing, prompt generation, or dataset-specific logic.

The current state of the project already gives us:

- dataset-specific preprocessing through `DatasetDriver`s in `src/data/datasets`
- a canonical row contract in `src/data/export/row_contract.py`
- validated per-dataset CSV manifests in `output/csv`
- a contract-based prompt system in `src/data/prompts`

Because of that, the fine-tuning layer should not recreate prompt formatting or dataset adaptation logic. Those responsibilities already belong to the data pipeline.

## Core Decision

`src/minim` should consume the outputs of dataset drivers, not raw datasets.

That means:

- `src/data` owns raw dataset ingestion, preprocessing, prompt generation, and CSV export
- `src/minim` owns experiment assembly, training orchestration, checkpoint management, generation, and evaluation

This removes the need for a `prompt_adapter.py` inside `src/minim`, because prompt normalization is now part of the dataset-driver contract and the prompt contract in `src/data/prompts`.

## New Boundary

The stable handoff between both layers is:

- processed images under `output/images/<dataset>/`
- validated manifests under `output/csv/<dataset>_minim.csv`

`src/minim` should treat those files as its training-ready input product.

## Proposed `src/minim` Structure

```text
src/minim/
  __init__.py
  cli.py
  constants.py
  types.py
  datasets/
    __init__.py
    manifest_registry.py
    dataset_catalog.py
  configs/
    __init__.py
    paths.py
    training_config.py
    serializers.py
  training/
    __init__.py
    launcher.py
    checkpointing.py
    logging.py
  generation/
    __init__.py
    sampler.py
  evaluation/
    __init__.py
    report.py
    metrics.py
  experiments/
    __init__.py
    registry.py
    naming.py
```

## What Each Part Should Do

### `src/minim/cli.py`

One user-facing entrypoint with commands such as:

- `prepare`: combine one or more dataset manifests into a training manifest
- `train`: launch model fine-tuning
- `generate`: sample images from a trained checkpoint
- `evaluate`: run evaluation and write reports

Example:

```bash
python -m minim.cli prepare --datasets acdc ukbb
python -m minim.cli train --experiment cardiac_sd15_v1
python -m minim.cli generate --experiment cardiac_sd15_v1 --checkpoint latest
python -m minim.cli evaluate --experiment cardiac_sd15_v1
```

### `src/minim/datasets/manifest_registry.py`

This module should resolve and validate the dataset manifests produced by `src/data`.

Responsibilities:

- locate `output/csv/<dataset>_minim.csv`
- verify the dataset exists and is readable
- assemble combined manifests when training with multiple datasets
- keep provenance information such as which datasets were merged

This replaces the old idea of a generic `manifest_builder.py` that also needed to understand prompt adaptation. It only needs to work with already normalized manifests.

### `src/minim/datasets/dataset_catalog.py`

This module should define which datasets are available for fine-tuning and how they map to exported manifests.

For example:

- `acdc`
- `ukbb`
- later `mm`

The catalog should not know how to preprocess those datasets. It should only know how to reference their exported products.

### `src/minim/configs/training_config.py`

Central place for experiment configuration:

- model identifier
- image resolution
- batch size
- learning rate
- number of steps or epochs
- mixed precision settings
- seed
- output/checkpoint paths
- datasets or manifest path

This should be serializable so each experiment stores the exact config used.

### `src/minim/training/launcher.py`

This is the main coupling point to the actual fine-tuning framework.

Responsibilities:

- receive a finalized manifest path plus a training config
- translate repository config into the arguments expected by MINIM or the chosen backend
- launch training reproducibly
- return artifact locations such as checkpoints and logs

This module should not know anything about raw datasets, metadata, or prompt construction.

### `src/minim/training/checkpointing.py`

Small helper layer for:

- latest checkpoint resolution
- best checkpoint resolution
- checkpoint naming
- resume support

### `src/minim/training/logging.py`

Standardize:

- console progress
- per-experiment log files
- structured metadata for runs

This should mirror the feedback improvements already added to `src/data`.

### `src/minim/generation/sampler.py`

Encapsulate image generation from:

- prompt lists
- manifest rows
- experiment checkpoints

This allows generation logic to evolve independently of training launch logic.

### `src/minim/evaluation/report.py`

Produce human-readable evaluation outputs:

- run summary
- used checkpoint
- manifest used
- generated sample paths
- selected metrics

### `src/minim/evaluation/metrics.py`

Isolate numeric evaluation:

- prompt-image alignment metrics if available
- reconstruction or similarity proxies if relevant
- future clinical or segmentation-based downstream checks

## What Should Be Removed From The Old Proposal

### Remove `prompt_adapter.py`

This is no longer justified.

Why:

- prompt normalization already exists in `src/data/prompts`
- datasets already adapt metadata through `DatasetDriver.to_prompt_payload(...)`
- exported manifests already contain final prompt text

Adding `src/minim/datasets/prompt_adapter.py` would reintroduce duplicated responsibility and weaken the contract boundary.

### Avoid duplicating dataset logic in `src/minim`

`src/minim` should not:

- parse ACDC or UKBB metadata
- compute BMI, EF groups, disease labels, or prompt segments
- perform dataset-specific preprocessing
- know dataset-specific file naming conventions

Those belong to the driver layer in `src/data`.

## Recommended Training Flow

### Step 1: Data export

Run:

```bash
python -m data.run_pipeline --dataset acdc
python -m data.run_pipeline --dataset ukbb
```

Outputs:

- processed images in `output/images/...`
- per-dataset manifests in `output/csv/...`

### Step 2: Manifest preparation

Run:

```bash
python -m minim.cli prepare --datasets acdc ukbb
```

Outputs:

- one combined training manifest
- optional dataset-specific provenance file

### Step 3: Fine-tuning

Run:

```bash
python -m minim.cli train --experiment cardiac_sd15_v1
```

Inputs:

- prepared manifest
- serialized training config

Outputs:

- checkpoints
- logs
- experiment metadata

### Step 4: Generation and evaluation

Run:

```bash
python -m minim.cli generate --experiment cardiac_sd15_v1 --checkpoint latest
python -m minim.cli evaluate --experiment cardiac_sd15_v1
```

## Why This Is Better

This proposal fits the current architecture instead of fighting it.

Benefits:

- one stable contract boundary between `src/data` and `src/minim`
- no duplicated prompt logic
- no duplicated dataset adaptation logic
- easier onboarding for new datasets: add a new `DatasetDriver`, export a manifest, and `src/minim` can consume it
- cleaner testing because `src/minim` can be tested against manifests instead of raw medical datasets

## Suggested Implementation Order

1. Create `src/minim/configs/training_config.py` and `src/minim/configs/paths.py`.
2. Implement `src/minim/datasets/manifest_registry.py`.
3. Implement `src/minim/cli.py` with at least `prepare` and `train`.
4. Implement `src/minim/training/launcher.py`.
5. Add experiment metadata persistence and checkpoint resolution.
6. Add `generate` and `evaluate`.

## Minimal First Deliverable

The smallest useful first version is:

```text
src/minim/
  __init__.py
  cli.py
  configs/
    training_config.py
    paths.py
  datasets/
    manifest_registry.py
  training/
    launcher.py
```

With that, the repository would already support:

- exporting normalized manifests with `src/data`
- preparing a final training manifest with `src/minim`
- launching one reproducible fine-tuning experiment

That is the cleanest next step given the current driver-based architecture.
