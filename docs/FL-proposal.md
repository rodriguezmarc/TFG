# Federated Learning Proposal For MINIM

## Objective

After centralized fine-tuning of MINIM, the next step should be a **federated fine-tuning pipeline** in which multiple sites train collaboratively without sharing raw cardiac MRI images.

The practical target is:

- each hospital/site keeps its own preprocessed CMR images and metadata locally
- a central coordinator shares model weights or adapter weights
- each site performs local updates
- the coordinator aggregates updates and redistributes the global model

For this project, the most realistic first implementation is:

- **Flower** as the FL orchestration framework
- **LoRA-style or parameter-efficient fine-tuning** if MINIM can support it cleanly
- **FedAvg-style aggregation first**
- **FLoRA-inspired aggregation later** if heterogeneous clients or heterogeneous adapter ranks become important

## Recommendation

### Why Flower

Flower is the safest engineering choice for the first FL implementation because it is:

- actively maintained
- framework-agnostic
- designed to wrap existing PyTorch training code
- usable both in simulation and in multi-node deployment

That matters here because this repository already has a local data pipeline and will likely need:

- a simulated FL experiment first
- later a site-based deployment across separate machines

### Where FLoRA fits

`FLoRA` is a **federated LoRA aggregation method**, not a full general-purpose FL platform.

So the correct interpretation is:

- Flower = the infrastructure/framework
- FLoRA = a possible aggregation strategy for LoRA adapters

For the first version, I would **not** start by implementing full FLoRA directly unless MINIM fine-tuning is already adapter-based and the team specifically wants heterogeneous client configurations.

The lowest-risk roadmap is:

1. central MINIM fine-tuning works
2. federated MINIM with Flower + FedAvg works
3. federated adapter tuning works
4. only then evaluate FLoRA-style aggregation

## Federated Learning Target Architecture

```text
Central coordinator
  - initializes global MINIM model or adapters
  - selects participating clients
  - sends current global state
  - receives client updates
  - aggregates updates
  - evaluates and stores round checkpoints

Client site A
  - local ACDC/MM/UKBB partition
  - local MINIM training loop
  - returns updated weights or adapter parameters

Client site B
  - local partition
  - local MINIM training loop
  - returns updated weights or adapter parameters

Client site C
  - local partition
  - local MINIM training loop
  - returns updated weights or adapter parameters
```

## Data Partition Strategy

The federated setup should not split individual patients randomly across clients. Instead, each client should represent a realistic institution.

Suggested simulated clients:

- `client_acdc`
- `client_mm`
- `client_ukbb`

If only ACDC is available initially, create multiple synthetic clients by partitioning ACDC at patient level:

- `acdc_site_1`
- `acdc_site_2`
- `acdc_site_3`

To make the simulation realistic, use **non-IID partitions**:

- client 1 richer in healthy and HCM cases
- client 2 richer in DCM and reduced EF cases
- client 3 richer in infarction or obese cases

This is important because the main value of FL appears when site distributions differ.

## What Should Be Federated

There are two possible levels.

### Option A: Federate the full MINIM fine-tuning

Each client updates the full trainable model used in centralized MINIM fine-tuning.

Pros:

- simplest conceptual match to centralized training
- easiest baseline for correctness

Cons:

- high communication cost
- likely expensive in GPU memory and transfer time
- less practical for real multi-site deployment

### Option B: Federate only adapters or lightweight trainable blocks

Each client trains:

- LoRA adapters
- or another parameter-efficient fine-tuning module

Pros:

- much lower communication cost
- better fit for FL
- easier personalization later

Cons:

- depends on MINIM fine-tuning code supporting adapters cleanly
- aggregation becomes more method-dependent

### Recommendation

If centralized MINIM fine-tuning is currently implemented as full fine-tuning, start with:

- **federated full-model baseline**

Then move to:

- **federated adapter tuning**

If MINIM is already being adapted with LoRA or can be modified with limited effort, skip directly to:

- **federated adapter tuning**

## Proposed Repository Structure

I would add a separate package instead of mixing FL code into `src/minim` directly.

```text
src/
  data/
    ...
  minim/
    ...
  federated/
    __init__.py
    cli.py
    types.py
    constants.py
    partitioning/
      __init__.py
      clients.py
      non_iid.py
      manifests.py
    flower/
      __init__.py
      server.py
      client.py
      strategy.py
      simulation.py
    training/
      __init__.py
      minim_wrapper.py
      local_train.py
      local_eval.py
      checkpointing.py
    aggregation/
      __init__.py
      fedavg.py
      fedprox.py
      flora_like.py
    evaluation/
      __init__.py
      global_metrics.py
      client_metrics.py
      privacy_report.py
      report.py
```

## Responsibility Of Each Module

### `src/federated/cli.py`

Commands such as:

- `simulate`
- `train`
- `evaluate`
- `resume`

Examples:

```bash
python -m federated.cli simulate --experiment acdc_fl_baseline
python -m federated.cli train --experiment acdc_mm_ukbb_fl
python -m federated.cli evaluate --experiment acdc_mm_ukbb_fl
```

### `src/federated/partitioning/clients.py`

Defines which datasets or subsets belong to each client.

This module should:

- map local CSV manifests to client ids
- ensure patient-level isolation
- support dataset-based clients and synthetic site-based clients

### `src/federated/partitioning/non_iid.py`

Creates realistic skewed splits based on:

- pathology
- EF category
- age category
- BMI category
- sex

This is needed because homogeneous FL is too optimistic for medical deployment.

### `src/federated/partitioning/manifests.py`

Builds one local manifest per client, for example:

- `output/federated/manifests/client_acdc_train.csv`
- `output/federated/manifests/client_mm_train.csv`

### `src/federated/flower/server.py`

Defines the Flower server application:

- initialize global model state
- coordinate rounds
- aggregate updates
- save round checkpoints
- trigger global evaluation

### `src/federated/flower/client.py`

Defines the Flower client:

- load local MINIM dataset
- receive global parameters
- run local training epochs
- return local update
- run local evaluation if requested

### `src/federated/flower/strategy.py`

Hosts the aggregation strategy.

Suggested order:

1. `FedAvg`
2. `FedProx`
3. `FLoRA-like adapter aggregation`

`FedProx` is useful because medical clients are often heterogeneous in sample count and optimization behavior.

### `src/federated/flower/simulation.py`

Runs local multi-client simulation on one machine before real deployment.

This should be the first milestone because it avoids infrastructure complexity.

### `src/federated/training/minim_wrapper.py`

This is the key integration layer.

It should adapt the centralized MINIM fine-tuning code into methods like:

- `get_parameters()`
- `set_parameters()`
- `train_local()`
- `evaluate_local()`
- `generate_samples()`

This wrapper prevents Flower-specific logic from leaking into `src/minim`.

### `src/federated/training/local_train.py`

Implements the local client-side training loop:

- load local manifest
- train for `k` local epochs
- compute local training loss
- save optional local checkpoint

### `src/federated/training/local_eval.py`

Computes client-side evaluation:

- local loss
- optional prompt-condition consistency checks
- optional local sample generation

### `src/federated/aggregation/fedavg.py`

Standard weighted averaging by client sample count.

This should be the baseline and primary comparison point.

### `src/federated/aggregation/fedprox.py`

Adds a proximal term to reduce client drift.

This is often useful when distributions differ strongly across institutions.

### `src/federated/aggregation/flora_like.py`

This module should be clearly marked as **experimental**.

Use it only if the training setup is adapter-based.

Its role would be:

- aggregate LoRA-style updates instead of full weights
- support heterogeneous adapter capacity across clients
- preserve a clean boundary between global transferable knowledge and local client-specific adaptation

I call it `flora_like.py` instead of `flora.py` on purpose, because the exact paper method may need adaptation for diffusion/MINIM rather than a direct copy.

## Implementation Plan

### Phase 1: Simulated Federated Baseline

Goal:

- prove that federated training runs end-to-end on local simulated clients

Work:

1. finish centralized MINIM fine-tuning
2. define `src/federated/training/minim_wrapper.py`
3. create patient-level client partitions
4. implement Flower simulation
5. implement FedAvg baseline

Deliverable:

- one federated training run over simulated clients
- round checkpoints
- logs of local and global losses

### Phase 2: Federated Evaluation

Goal:

- compare centralized vs federated generation quality

Work:

1. generate synthetic images from the global federated model
2. compute:
   FID
   IS
   MS-SSIM
3. compare against centralized MINIM fine-tuning
4. compare against client-local models if useful

Deliverable:

- one benchmark table for centralized vs federated

### Phase 3: Heterogeneity-Robust FL

Goal:

- improve performance under non-IID client distributions

Work:

1. add FedProx
2. add client weighting experiments
3. test partial client participation
4. test unbalanced client sizes

Deliverable:

- robustness analysis under realistic site heterogeneity

### Phase 4: Adapter-Based Federated Tuning

Goal:

- reduce communication and improve scalability

Work:

1. adapt MINIM fine-tuning to LoRA or another PEFT method
2. federate only adapter parameters
3. implement `flora_like.py`
4. compare against full-model FedAvg

Deliverable:

- communication-efficient federated MINIM variant

## Evaluation Protocol

The federated model should be evaluated at three levels.

### 1. Global Image Quality

Same metrics already planned for centralized MINIM:

- FID
- Inception Score
- MS-SSIM

These should be measured on the global federated model after selected rounds.

### 2. Client Fairness

Do not report only one pooled score. Also report per-client performance:

- FID per client
- IS per client
- MS-SSIM per client

This matters because FL can improve average performance while failing badly on small or underrepresented sites.

### 3. Federated Optimization Metrics

Track:

- round number
- selected clients per round
- local epochs
- client sample counts
- communication volume
- convergence stability
- wall-clock time

## Expected Results

The likely outcomes are:

### Expected positive outcomes

- federated MINIM should approach centralized fine-tuning quality if clients are reasonably aligned
- FL should outperform purely local client-specific models
- adapter-based FL should be much cheaper to communicate than full-model FL
- non-IID-aware methods such as FedProx or FLoRA-like aggregation may improve stability over plain FedAvg

### Expected negative outcomes

- federated training will probably underperform centralized training at first
- small clients may suffer from unstable local updates
- strong non-IID pathology skews may hurt global generation quality
- communication cost may be prohibitive if full MINIM weights are exchanged every round

That is normal. The first success criterion should not be “beat centralized,” but:

- demonstrate privacy-preserving collaborative fine-tuning
- quantify the performance gap
- identify whether adapter-based FL closes that gap

## Risks

### Technical risk

MINIM’s training scripts may not be easy to wrap for repeated local training and parameter exchange.

Mitigation:

- isolate all coupling in `minim_wrapper.py`

### Compute risk

Full-model federated diffusion fine-tuning may be too expensive.

Mitigation:

- start with small simulations
- reduce image count
- reduce rounds
- switch to adapter-based training

### Evaluation risk

FID and IS are imperfect for medical images.

Mitigation:

- still report them for comparability
- add qualitative review
- optionally add domain-specific feature metrics later

### Privacy risk

Basic FL is not automatically private against all attacks.

Mitigation:

- in the first version, describe it as “data-local collaborative training,” not as a complete privacy guarantee
- later add secure aggregation and differential privacy if needed

## Concrete Deliverables

The implementation should produce:

- `src/federated/...` package
- client-specific manifests
- Flower simulation scripts
- global federated checkpoints
- generated images per round or per selected checkpoints
- metric tables comparing centralized, local-only, and federated models
- an experiment report summarizing convergence and quality

## Minimal Viable FL Experiment

The smallest meaningful experiment is:

1. centralized MINIM fine-tuning on ACDC works
2. split ACDC into 3 synthetic clients at patient level
3. run Flower simulation with FedAvg
4. train for a small number of rounds
5. generate samples from the global model
6. compare against centralized MINIM with FID, IS, MS-SSIM

If that works, then the project has a valid federated baseline.

## Final Recommendation

Implement FL in two layers:

- `src/minim` remains the centralized model-training layer
- `src/federated` becomes the FL orchestration layer

Use:

- **Flower first**
- **FedAvg first**
- **FedProx second**
- **FLoRA-inspired aggregation later if adapters are used**

This is the most defensible engineering path because it minimizes risk, preserves modularity, and gives a clear comparison chain:

- local only
- centralized MINIM
- federated MINIM with FedAvg
- federated MINIM with stronger aggregation

## Notes On Tool Choice

At the time of writing:

- Flower is a maintained federated learning framework suitable for wrapping existing training code
- FLoRA is a 2024 federated LoRA aggregation method, useful as inspiration for adapter aggregation rather than as the whole platform

So for this project, Flower should be the implementation backbone and FLoRA should be treated as a later algorithmic extension.
