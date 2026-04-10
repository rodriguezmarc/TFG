# Roadmap

## Purpose

This roadmap reflects the current architecture of the repository after the migration to:

- dataset drivers in `src/data/datasets`
- prompt generation by explicit contract
- a canonical row contract before CSV export
- a clean handoff from `src/data` to a future `src/minim`

The goal now is not to redesign `src/data` again, but to build the remaining layers on top of this stabilized base.

## Current Baseline

The repository already has:

- driver-based dataset ingestion and export in `src/data/datasets`
- reusable cardiac driver flow in `BaseCardiacDatasetDriver`
- ACDC and UKBB integrated through `DatasetDriver`s
- prompt generation based on `PromptPayload` and `PromptCapabilities`
- validated MINIM-style CSV export in `src/data/export`
- test coverage for drivers, preprocessing, prompts, export, and shared utilities

The repository does not yet have:

- a completed `src/minim` package
- a finalized centralized fine-tuning launcher
- experiment tracking and checkpoint management
- generation and evaluation commands for the training layer
- a `src/federated` package integrated with the new driver-based data product

## Execution Principles

- Keep `src/data` responsible for raw-data handling, preprocessing, prompt generation, and manifest export.
- Keep `src/minim` responsible for training orchestration, checkpointing, generation, and evaluation.
- Never duplicate prompt adaptation in `src/minim`.
- Treat dataset drivers as the only dataset-specific integration point.
- Preserve the manifest contract as the stable interface between `src/data` and `src/minim`.

## Workstreams

There are now two main workstreams:

1. Centralized fine-tuning layer
2. Federated fine-tuning layer

The dependency order remains:

```text
src/data drivers and manifests
  -> Centralized fine-tuning
  -> Federated fine-tuning
```

## Phase 1: Consolidate The Driver-Based Data Product

### Goal

Freeze the current `src/data` output as the official product consumed by training.

### Tasks

1. Keep the driver contract stable.
2. Keep the prompt contract stable.
3. Keep the row contract stable.
4. Align remaining docs with the new architecture.

### Deliverables

- stable `DatasetDriver` API
- stable prompt contract
- stable row contract
- updated documentation

### Exit Criteria

- no architectural ambiguity between functions and drivers
- no documentation referring to deprecated prompt adaptation inside training
- manifests remain valid and reproducible

## Phase 2: Create The Minimal `src/minim` Training Layer

### Goal

Introduce the smallest training package that can consume exported manifests and launch one reproducible fine-tuning run.

### Required structure

```text
src/minim/
  __init__.py
  cli.py
  configs/
    __init__.py
    training_config.py
    paths.py
  datasets/
    __init__.py
    manifest_registry.py
    dataset_catalog.py
  training/
    __init__.py
    launcher.py
```

### Tasks

1. Define training config objects.
2. Resolve exported manifests from `output/csv`.
3. Implement a CLI with at least `prepare` and `train`.
4. Implement the launcher boundary to the actual fine-tuning backend.

### Deliverables

- non-empty `src/minim` package
- a manifest preparation command
- a first training launch command

### Exit Criteria

- `python -m minim.cli --help` works
- `python -m minim.cli prepare ...` works
- `python -m minim.cli train ...` has a stable interface

## Phase 3: Implement Manifest Preparation For Training

### Goal

Prepare one or more exported dataset CSVs for centralized fine-tuning.

### Tasks

1. Read per-dataset manifests from `output/csv`.
2. Support single-dataset and multi-dataset preparation.
3. Preserve provenance of which datasets were used.
4. Export one final training manifest for `src/minim`.

### Deliverables

- `src/minim/datasets/manifest_registry.py`
- combined training manifest under `output/minim/manifests/`

### Exit Criteria

- one command can merge `acdc`, `ukbb`, and future datasets without touching `src/data`

## Phase 4: Implement Centralized Fine-Tuning Launch

### Goal

Run one reproducible training job from the repository using prepared manifests.

### Tasks

1. Freeze the training config schema.
2. Implement the launcher.
3. Capture logs, configs, and checkpoint paths per experiment.
4. Make resume behavior explicit.

### Deliverables

- training launcher
- checkpoint directory convention
- experiment metadata persistence

### Exit Criteria

- one command launches fine-tuning reproducibly
- experiment config is stored
- checkpoints are organized and recoverable

## Phase 5: Add Generation And Evaluation

### Goal

Make the centralized training layer usable end to end.

### Tasks

1. Add generation from experiment checkpoints.
2. Add evaluation reports.
3. Standardize output directories for samples and reports.

### Deliverables

- `src/minim/generation`
- `src/minim/evaluation`
- repeatable post-training workflow

### Exit Criteria

- generation works from a trained checkpoint
- evaluation writes a report for the experiment

## Phase 6: Prepare The Federated Layer

### Goal

Build the centralized path first, then add the federated layer on top of it without breaking modular boundaries.

### Tasks

1. Reuse prepared manifests as client inputs.
2. Keep federated orchestration outside `src/minim`.
3. Wrap centralized training entrypoints instead of duplicating them.

### Deliverables

- a future `src/federated` package with clear boundaries

### Exit Criteria

- centralized and federated code can evolve independently

## What Is Explicitly No Longer Planned

The following ideas are now deprecated because they conflict with the current architecture:

- putting prompt adaptation inside `src/minim`
- adding `prompt_adapter.py` under `src/minim/datasets`
- duplicating dataset metadata normalization outside drivers
- letting training code understand raw dataset layouts

## Short-Term Priority

The next concrete implementation target should be:

1. create `src/minim/configs/training_config.py`
2. create `src/minim/datasets/manifest_registry.py`
3. create `src/minim/cli.py`
4. create `src/minim/training/launcher.py`

That is the shortest path from the current repository state to a working centralized fine-tuning baseline.
