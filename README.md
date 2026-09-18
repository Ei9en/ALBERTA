# ALBERTA

**Active Learning, Behavioral-cloning and Evolution-based Reinforcement for Thrifty Agents**

> Sparse expert supervision for league-based reinforcement learning, with response-aware bi-objective acquisition.

ALBERTA is a research project studying how a very small amount of human expert supervision can be allocated during reinforcement learning.

The experimental environment is **Atomic Chess**. A neural agent is first pretrained by behavioral cloning, then trained through league-based self-play with PPO. Starting from a shared RL checkpoint, different branches receive the same expert-intervention budget but differ in how annotated positions are selected.

The main experiment studies a response-aware acquisition strategy: instead of selecting positions only from uncertainty statistics, ALBERTA measures how expert annotations locally change the learner and models two response signals:

- policy displacement, measured by KL divergence;
- absolute value displacement, `|ΔV|`.

Predicted responses are then used as a **bi-objective Pareto acquisition criterion**.

## Overview

The experimental pipeline is:

```text
Lichess Atomic Chess games
        │
        ▼
Behavioral Cloning
        │
        ▼
League-based PPO self-play
        │
        ▼
      RL10
        │
        ├───────────────┬──────────────────┐
        │               │                  │
        ▼               ▼                  ▼
   PPO only       Random acquisition      Pareto acquisition 
                        │                  │
                        │          200 calibration probes
                        │                  │
                        │          local learner response
                        │             (ΔKL, |ΔV|)
                        │                  │
                        │          response prediction
                        │                  │
                        │          2-D Pareto selection
                        │                  │
                        └────────┬─────────┘
                                 │
                      121 expert interventions
                                 │
                                 ▼
                               RL20
```

The intervention budget is approximately **0.01%** of the ~1.2M self-play observations collected before the branch point: **121 expert annotations per supervised treatment**.

The Pareto treatment additionally uses **200 expert-annotated calibration probes** to learn the local response model. These calibration annotations are separate from the 121-position intervention budget.

## Main result

Three final policies are compared:

| Policy | Training after RL10 |
| --- | --- |
| **RL** | PPO self-play only |
| **Random** | PPO + 121 randomly selected expert interventions |
| **Pareto** | PPO + 121 response-aware Pareto-selected expert interventions |

Each pair was evaluated over 500 games at four action-sampling temperatures.

### Pareto vs. PPO-only RL

| Temperature | Pareto score |
| ---: | ---: |
| 0.25 | **63.8%** |
| 0.50 | **58.3%** |
| 1.00 | **53.7%** |
| 2.00 | **55.1%** |

Pareto finishes ahead of the PPO-only control at all four tested temperatures in this controlled training run.

### Pareto vs. Random supervision

| Temperature | Pareto score |
| ---: | ---: |
| 0.25 | **77.4%** |
| 0.50 | **62.3%** |
| 1.00 | **56.8%** |
| 2.00 | **50.3%** |

The difference is strongly temperature-dependent and becomes small at high sampling temperature.

These results are from **one controlled training seed (42)**. The 500-game evaluations reduce match-level sampling noise, but they are not independent training replications. The experiment should therefore be interpreted as a proof of concept rather than evidence of multi-seed or cross-domain robustness.

## Response-aware acquisition

For each calibration annotation, ALBERTA starts from the same RL10 learner and performs one controlled Oracle-only optimization step.

The learner is reset before every intervention.

Two local response magnitudes are measured:

```text
ΔKL   policy displacement after the expert update
|ΔV|  absolute value-function displacement
```

These quantities measure **how much the learner changes**, not whether the local update is beneficial.

An ExtraTrees multi-output regressor predicts both responses from contemporary learner-dependent features:

```text
H   policy entropy
U   historical-league value disagreement
HU  interaction between H and U
```

The historical uncertainty estimator uses trained historical RL critics and the current critic; behavioral-cloning models with untrained value heads are excluded.

Candidate positions are ranked using exact two-dimensional Pareto non-dominance and crowding over predicted `(ΔKL, |ΔV|)`. The final intervention queue contains 121 positions from the first Pareto front.

The downstream tournament, rather than the local response magnitude itself, determines whether the resulting training intervention was useful.

## Experimental controls

All final branches share:

- the same RL10 branch point;
- the same neural architecture;
- the same PPO/self-play setup;
- the same master seed (`42`);
- the same 121-position intervention budget for supervised treatments.

Internal trajectory checks at epochs 10, 15 and 20 show substantial learning in all three branches. The final comparison therefore does not rely on a regressing PPO control or a collapsed Random branch.

No completed training comparison against the earlier uncertainty-score acquisition condition is reported. Likewise, the project does not currently include multi-seed training replication, policy-only/value-only response ablations, scalarization-vs-Pareto ablations, or annotation-budget sweeps.

## Repository structure

```text
AL/
├── HMI/                         Human Oracle annotation interface
├── pareto/
│   ├── sample_pareto_probe.py  Calibration-probe sampling
│   ├── mesure_local_response.py
│   └── select_pareto.py        Response model + Pareto acquisition
├── estimate_active_learning_weights.py
└── seed_oracle_queue.py

analysis/                        Optional scientific diagnostics
evaluation/                      BC/RL tournaments and evaluations
making_dataset/                  Lichess → BC dataset pipeline
src/                             Models, agents, self-play and RL utilities
training/
├── train_bc.py
├── train_rl.py
└── train_al.py

data/
├── queue/                       Versioned expert annotation queues
└── pareto/                      Versioned calibration responses/results

lichess_bot/                     Lichess deployment integration
```

Large datasets and model checkpoints are intentionally not stored in Git.

## Models

The behavioral-cloning policy uses a residual convolutional network over an `8 × 8` Atomic Chess board representation.

The final BC architecture uses:

```text
Input channels: 19
Residual channels: 32
Residual blocks: 4
Action space: 20,160
Parameters: ~11.5M
```

RL wraps the pretrained policy backbone in an Actor-Critic model and trains it through league-based PPO self-play.

The action vocabulary is fixed across BC, RL, datasets and checkpoints for compatibility.

## Data

Behavioral cloning is trained from rated Atomic Chess games from the Lichess database.

The local pipeline produces:

```text
data/processed/positions_2300.jsonl
data/processed/positions_2300_bc.jsonl
```

The final BC dataset is approximately 340 MB and is therefore not committed to Git.

It can be constructed using:

```bash
python making_dataset/extract_positions.py
python making_dataset/build_bc_dataset.py
```

Historical self-play statistics used by the active-learning analyses are likewise kept outside Git because of their size.

The small expert-annotation queues and Pareto calibration artifacts used by the final experiments are versioned under `data/queue/` and `data/pareto/`.

## Active-learning artifacts

The repository includes the exact small artifacts defining the supervision experiments:

```text
data/queue/oracle_queue_1-10_random.jsonl
data/queue/oracle_queue_1-10_AL.jsonl
data/queue/oracle_queue_1-10_pareto_probe.jsonl
data/queue/oracle_queue_dynamic_pareto_10.jsonl

data/pareto/pareto_local_responses.jsonl
data/pareto/pareto_selection_10.json
```

`oracle_queue_1-10_pareto_probe.jsonl` contains the 200 manually annotated calibration interventions.

`oracle_queue_dynamic_pareto_10.jsonl` contains the final 121-position Pareto intervention queue.

## Pareto pipeline

The final response-aware acquisition pipeline consists of exactly three stages:

```bash
python AL/pareto/sample_pareto_probe.py
python AL/pareto/mesure_local_response.py
python AL/pareto/select_pareto.py
```

Conceptually:

```text
historical RL states
        │
        ▼
200 calibration positions
        │
        ▼
human expert annotations
        │
        ▼
controlled one-step learner responses
        │
        ├── ΔKL
        └── |ΔV|
        │
        ▼
ExtraTrees response model
        │
        ▼
refresh H/U/HU at RL10
        │
        ▼
predict responses over candidate pool
        │
        ▼
2-D Pareto selection
        │
        ▼
121 expert interventions
```

## Training

Behavioral cloning:

```bash
python training/train_bc.py --help
```

League PPO:

```bash
python training/train_rl.py --help
```

Sparse expert-supervised training:

```bash
python training/train_al.py --help
```

The training scripts expose their experimental configuration through their command-line interfaces. Checkpoints are written locally under `checkpoints/` and are ignored by Git.

## Evaluation

RL/AL checkpoints can be evaluated with:

```bash
python evaluation/tournament_rl.py --help
```

BC checkpoints can be evaluated with:

```bash
python evaluation/tournament_bc.py --help
```

The reported final experiment uses action temperatures:

```text
0.25, 0.5, 1.0, 2.0
```

with 500 games for each pairwise matchup at each temperature.

## Optional analyses

`analysis/` contains diagnostics developed during the project, including:

- uncertainty progression and reward-signal analysis;
- annotation-response analysis;
- queue-composition analysis;
- calibration-probe coverage;
- trained-policy shift;
- PPO/Oracle gradient-alignment diagnostics;
- reference-gradient stability.

These scripts are supplementary diagnostics and are not all required to reproduce the main training experiment.

## Scope

ALBERTA is a single-domain research prototype built around Atomic Chess.

The project investigates whether **learner response to supervision** can provide a useful acquisition signal under an extremely sparse expert-intervention budget.

The current evidence supports this idea in one controlled training run. It does not establish that the method is universally superior to uncertainty sampling, nor that the observed effect generalizes across random seeds, RL algorithms, games, or other domains.

## Acknowledgements

Atomic Chess game data is derived from the public Lichess database.

The Lichess deployment integration builds on the `lichess-bot` project; see the files and license under `lichess_bot/` for upstream attribution.

## Third-party software and assets

ALBERTA includes or adapts third-party components that retain their
respective licenses:

- `lichess_bot/` is based on the open-source `lichess-bot` project and is
  distributed under the GNU Affero General Public License v3. See
  `lichess_bot/LICENSE`.

- Chess-piece SVGs under `AL/HMI/assets/pieces/` use the Cburnett piece set
  by Colin M. L. Burnett, obtained from the Lichess source repository.
  The Cburnett piece set is distributed under the GNU GPL v2 or later.

These third-party components are not relicensed under ALBERTA's project
license.