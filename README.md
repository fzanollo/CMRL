#  CMRL: Controller Mimicking Reinforcement Learning

CMRL trains a Deep Q-Network agent as a reachability controller on a pre-labeled monolithic plant, fully decoupling training from the costly On-the-fly Directed Controller Synthesis (OTF-DCS) procedure. The resulting heuristic achieves competitive synthesis performance on unseen Sokoban instances at a fraction of the training cost of prior RL-based approaches.

## Repository Layout

- `environment/`
  - `environment.py`: main `gymnasium.Env` implementation.
  - `observation_spaces.py`: observation encodings.
- `utils/`
  - `sokoban_generator.py`: random board + `.lts` generation.
  - `template.txt`: LTS template used by the generator.
  - `utils.py`, `model_utils.py`: plotting and helper scripts.
  - `comp_state_info.py`, `lts_to_png.py`: additional utilities.
  - `sprites/`: visual assets for Sokoban rendering.
- `train_data/`
  - `lts/`: input benchmark `.lts` files used for training.
  - `sokoban-4-1/` … `sokoban-7-1/`: parsed feature/action CSVs per problem size.
- `generated_sokoban_specs/`
  - `sokoban-{size}-1/`: generated Sokoban `.lts` instances grouped by board size.
- `rl_results/`
  - Results from the 5 independent CMRL training runs (`run_1/` … `run_5/`), each containing trained agent checkpoints and per-problem synthesis results. Also includes Delgado et al.'s prior RL baseline results for direct comparison. Runtime summary CSVs (`runtime_totals_per_run.csv`, `runtime_totals_summary.csv`) aggregate wall-clock times across all runs.
- `ra_results/`
  - Results obtained using the Random Action (RA) heuristic, one CSV per problem size.
- `monolithic_results/`
  - Results from the monolithic (exhaustive) synthesis baseline, one CSV per problem size.
- Top-level scripts:
  - `DQNAgent.py`: DQN agent definition.
  - `get_raw_data.py`: generates raw features and controller artifacts.
  - `collect_actions_info.py`: aggregates controllable/uncontrollable actions per problem.
  - `training.py`: trains DQN agents.
  - `run_mtsa.py`: runs comparative synthesis experiments.
  - `human_env_vs_agent.py`: interactive environment for manual play / agent comparison.

## Requirements

From `requirements.txt`:
- `torch`, `numpy`, `pandas`, `matplotlib`, `gymnasium`, `networkx`, `onnx`, `scipy`, `tqdm`, `ipython`

Additionally, MTSA execution requires Java and `mtsa-1.0-SNAPSHOT.jar` (already present in project root).

## Setup

```bash
pip install -r requirements.txt
```

## Typical Workflow

1. **Generate raw features and controller artifacts for the instances you will be using to train:**

```bash
python get_raw_data.py
```

2. **Aggregate available controllable/uncontrollable actions per problem:**

```bash
python collect_actions_info.py
```

3. **Train DQN agents:**

```bash
python training.py
```

4. **Run synthesis experiments for RA and RL heuristics:**

You can either use newly trained models or reuse the pre-trained models stored in `rl_results/run_*/trained_agents/`.

To use pre-trained models with `run_mtsa.py`:

- Convert `.pth` checkpoints to `.onnx` inside the selected `trained_agents` folder:

```bash
python convert_pth_to_onnx.py --models-dir rl_results/run_1/trained_agents
```

- Run with explicit configuration:

```bash
python run_mtsa.py --lts-dir generated_sokoban_specs --trained-agents-dir rl_results/run_1/trained_agents --timeout-seconds 600
```

We do not provide Delgado et al.'s training pipeline in this repository; it is available at [Learning-Synthesis](https://github.com/tdelgado00/Learning-Synthesis).

`run_mtsa.py` is configurable from outside with:

- `--lts-dir`: base directory containing `{problem}/{instance}.lts`
- `--trained-agents-dir`: base directory containing model folders and `prev_rl/`
- `--timeout-seconds`: timeout per synthesis call

Environment variable alternatives:

- `CMRL_LTS_DIR`
- `CMRL_TRAINED_AGENTS_DIR`
- `CMRL_TIMEOUT_SECONDS`

## Sokoban Spec Generation

`utils/sokoban_generator.py` generates Sokoban `.lts` instances from randomly created in-memory boards.

### Output Structure

Generated files are saved under `generated_sokoban_specs/sokoban-{size}-1/` with the naming format:

```
sokoban-{size}-1_{obstacles}obstacles_{id}.lts
```

Example: `generated_sokoban_specs/sokoban-5-1/sokoban-5-1_3obstacles_7.lts`

### Board Conventions

- Board is square (`size × size`)
- Fixed player start at `(1,1)` (top-left)
- One brick (`B`), one goal (`F`), and `N` obstacles (`O`)
- Empty cells are `-`
- Board diagram is printed to console and embedded in the generated `.lts` file

### Uniqueness Guarantee

For each `(size, obstacle_count)` batch, generated configurations are unique.
The generator tracks board signatures and skips duplicates.

With fixed player cell `(1,1)`, the maximum number of unique configurations is:

`max_unique = A × (A − 1) × C(A − 2, obstacles)`

where `A = size² − 1` (all non-player cells).

### Usage

Run the default batch (sizes `4..7`, obstacles `1..4`, 10 configurations each):

```bash
python utils/sokoban_generator.py
```

Custom generation:

```bash
python utils/sokoban_generator.py 6 20 --obstacles 3 --seed 42
```

Parameters:
- `board_size` (positional): `N` for an `N × N` board
- `configurations` (positional): number of unique configs to generate
- `--obstacles`: number of obstacles per board
- `--seed`: optional seed for reproducibility

## Notes

- Several experiment scripts contain hardcoded problem/scenario lists — adjust them before large runs.
- The sprites used for the visual representation of the Sokoban maps were obtained from the *Sokoban* asset pack by [Kenney](https://kenney.nl/assets/sokoban).