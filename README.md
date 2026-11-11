# Five-reservoir HAPPO and MAPPO

Daily operation of the WDD, BHT, XLD, XJB and THR cascade using a specialized
feed-forward HARL implementation. The objective combines hydropower generation
with a soft ecological-flow penalty. Water balance, storage envelopes, outlet
capacities and episode-end storage equality remain hard physical constraints.

## Run

Install the dependencies declared in `setup.py` into a suitable Python environment:

```bash
python -m pip install -e .
```

Validate data, episode windows and observations without training:

```bash
python -B examples/train_cascade.py --check-data-only
```

Train HAPPO and MAPPO with paired seeds and three comparison figures:

```bash
python -B examples/train_cascade_compare.py \
  --episodes 300 --seeds 1 2 3 --gamma 0.995 --eval-interval 1 --diagnostics \
  --plot-output runs/ecological_comparison
```

The default comparison runs both algorithms; select `--algorithms happo` or
`--algorithms mappo` to run one. `examples/train_cascade.py` remains the HAPPO
entry point. The episode budget applies separately to each algorithm and seed.
`--no-plot` disables figures. `--plot-python PATH` selects another interpreter
with Matplotlib. Existing output files are never overwritten.

By default results stay in memory. Add a fresh `--episode-log` path to retain
JSONL summaries and per-seed daily CSVs. For comparison runs that path is a stem:
separate `_happo.jsonl` and `_mappo.jsonl` logs are created. Add
`--save-checkpoints` to save final policies, or `--checkpoint-interval N` to also
save every N updates. Checkpoints support inference replay, not training resume.

## Data, observations and physical model

Inputs are in `data/cascade_reservoir_new/`. Actual historical daily external
inflows cover 1965–2024. BHT/XLD/XJB/THR columns are incremental interval inflows;
upstream executed release is added by the physical cascade. Releases propagate
upstream to downstream within one daily transition, with zero routing delay.

`five_reservoir_happo_input_bounds_1965_2024.csv` supplies the permitted next-state
storage and level for each action date. Its row for date t constrains the state
after action t; no extra date shift is applied. Previous executed bounds constrain
the current state, including across source-year joins.

`terminal_storage_equal_initial: true` requires every reservoir to finish the
full episode at its own episode-start storage. Backward reachability uses known
scenario inflows, outlet capacities and daily storage envelopes. Coupled
cumulative-storage bounds keep this target reachable while commands are executed.
There are no intervening annual equality targets or storage resets. Final storage
is calculated from releases and water balance and checked within numerical
water-accounting tolerance; it is never overwritten to meet the target.

Actor observations contain local and immediate-neighbor storage, current inflow,
yesterday's release, downstream headroom, calendar/time features and daily bounds.
The supplied `five_reservoir_next7d_mean_inflow_forecast_1965_2024.csv` contributes
one additional local feature: the date-t forecast of next-seven-day mean inflow,
standardized using actual-inflow statistics from training years only. Forecasts
are read from the supplied file, including at nonconsecutive held-out year joins;
they are not recomputed from future actual inflows. Actors receive 16 features
with daily limits, or 12 without them. Centralized critic states remain 38 or 18
features and do not receive the forecast feature.

For categorical action k among K commands:

```text
u = k / (K - 1)
release = feasible_low + u² * (feasible_high - feasible_low)
```

Feasible release intervals enforce storage bounds, outlet capacities,
downstream compatibility and terminal reachability. Physically forced intervals
mask actor learning and contribute a unit HAPPO correction factor. The default
100-command grid can be changed with `--action-bins`.

Generation uses turbine flow, installed capacity and net head. The default
`storage_sensitivity` head model depends on mean start/end forebay elevation and
supplied slope/offset assumptions; this is a sensitivity model. `--head-model
fixed` selects typical constant heads.

## Ecological flow and rewards

`reservoir_service_requirements.csv` supplies only `ecological_flow_m3s`, in
WDD/BHT/XLD/XJB/THR order. The requirement applies to total reservoir release,
including turbine flow and spill, on every active day. Zero thresholds are
inactive. `--ecological-flow-scale` scales reference CSV values and records the
effective thresholds and input hash. Ecological compliance is a soft objective;
it does not replace commands or impose a hard release floor.

```text
relative_flow_deficit_i = max(threshold_i - release_i, 0) / threshold_i
local_i = power_i / installed_capacity_i
          - ecological_flow_weight * relative_flow_deficit_i
global = sum(power_i) / sum(installed_capacity_i)
mixed_i = 0.5 * local_i + 0.5 * global
system = 0.5 * sum(local_i) + 0.5 * global
```

The deficit is zero for inactive thresholds and tolerance-filtered for numerical
roundoff. Local penalties use each reservoir's own relative deficit. The global
term measures generation only and enters system accounting once. The legacy
team metric is normalized total generation minus the mean ecological deficit
over the fixed active reservoir set. It remains separate from the learning
reward and from the sum of mixed rewards. `--no-reward-penalties` selects a
generation-only objective while physical ecological counts remain observable.

## Algorithms and training

The main settings are in `harl/configs/algos_cfgs/happo_cascade.yaml` and physical
settings in `harl/configs/envs_cfgs/cascade_reservoir.yaml`. Defaults are six
process-parallel episodes per update, 1,825 daily transitions per episode,
300 episodes per seed, gamma 0.995, GAE lambda 0.95 and five actor/critic epochs.
The default therefore performs 50 updates per seed. Leap days remain in the data;
the fixed horizon need not end on December 31.

HAPPO uses five independent actors and one scalar centralized team critic.
All actors receive one team GAE advantage, normalized once across rollout rows.
Actors update sequentially with accumulated likelihood corrections for previously
updated actors. PPO clips the current actor ratio. Control masks zero forced
rows, and losses retain a mean over the entire rollout.

MAPPO uses one shared actor and optimizer. Five lightweight views append a
five-way one-hot reservoir identity, giving 21 actor inputs with daily limits.
Own mixed-return advantages are pooled in one joint training batch. One
centralized critic produces five mixed-return values plus a legacy accounting
value. MAPPO uses unit factors and no sequential likelihood correction. Its
sharing, reward targets and critic layout differ from HAPPO, so comparisons do
not isolate a single algorithmic difference.

Actor and critic MLPs use hidden widths `[64, 128, 256, 256]`, ReLU, hidden
LayerNorm, categorical action distributions, Adam and gradient clipping. GAE and
actor advantages use rollout ValueNorm statistics. The normalizer updates once
from full-rollout critic targets after actor training and stays fixed through
critic epochs. Training uses CPU execution.

`--happo-credit-assignment counterfactual` optionally replaces HAPPO actor GAE
with frozen one-step, model-assisted counterfactual credit. It enumerates commands
under the same full physical constraints and ecological reward, holding other
commands fixed. The scalar team critic retains its GAE targets and sequential
HAPPO corrections remain. Default credit assignment is `team_gae`.

## Sampling, evaluation and plots

Five source years are held out by default: 1998, 1999, 1970, 1997 and 2006.
Training windows contain training years only. Coverage-balanced sampling reduces
unequal year-day exposure; `--window-sampling uniform` restores uniform draws.
Storage carries across source-year boundaries within each episode.

Periodic evaluation runs on three fixed training-only scenarios before update
one, every configured interval and after the final update. It uses deterministic
policies and fresh environments without updating models, optimizers or RNG state.
Final held-out evaluation uses an ordered representative-year sequence and the
same episode horizon. Physical infeasibility is reported explicitly; failed
windows remain failures, with no completed aggregate substituted from survivors.

Exactly three plot artifacts are available: stochastic training return/generation,
fixed-scenario evaluator return, and a single-panel ecological-flow violation
rate. Means and sample standard deviations stay separate by algorithm. Violation
rates count violating active reservoir-days, independent of reward weights and
gamma. Ecological deficit volumes and generation are undiscounted; reported
returns retain gamma. Failed or missing evaluations leave gaps.

```bash
python -B examples/plot_cascade_parallel.py --episode-log runs/EXPERIMENT.jsonl
python -B examples/plot_cascade_compare.py \
  --happo-log runs/EXPERIMENT_happo.jsonl --mappo-log runs/EXPERIMENT_mappo.jsonl
```

Plot validators check scenario identities, data hashes, physical settings and
training budgets. Forecast input hashes identify current actor dimensions while
older logs without that input retain their original dimensions.

## Diagnostics and replay

`--diagnostics` reports parameter changes, entropy/KL, surrogate changes and
Monte Carlo/GAE fit on collected trajectories. `--trajectory-diagnostics` requires
an episode log and records physical dispatch, ecological deficits, reward
components, frozen values, GAE and policy factors per reservoir/day/lane. Global,
system and legacy scalar rewards repeat across five CSV rows; count each once
per day/lane when reconstructing totals.

```bash
python -B examples/replay_cascade_checkpoint.py \
  --checkpoint runs/EXPERIMENT_seed1_checkpoints/update_000050.pt \
  --source-years 2007 2008 2009 2010 2011 --output runs/EXPERIMENT_replay.csv
```

`--override-action DAY RESERVOIR INDEX` changes one command; later commands react
through the saved policy. Replay preserves the saved horizon, physical and data
hashes, forecast layout and terminal semantics. It reports generation, ecological
deficits, learning returns, spill and final storage without fitting models.

`analysis/dispatch_diagnostics/` reconstructs frozen trajectories and tests
same-state ecological release alternatives through the full cascade. Grid and
continuous witnesses retain terminal constraints. These are separate same-day
comparisons, not an executable multi-day intervention or evidence of optimality.

## Source map

| Path | Responsibility |
| --- | --- |
| `examples/train_cascade.py` | CLI, configuration and split validation |
| `examples/train_cascade_compare.py` | Paired HAPPO/MAPPO orchestration |
| `examples/train_cascade_parallel.py` | Seed runs, logging and evaluation |
| `harl/runners/` | Full-episode collection and algorithm update order |
| `harl/algorithms/`, `harl/models/`, `harl/common/` | PPO, critics, networks, buffers and ValueNorm |
| `harl/envs/cascade_reservoir/data_loader.py` | Historical data, forecasts and ecological inputs |
| `harl/envs/cascade_reservoir/reservoir_physics.py` | Water balance, generation and hard feasibility |
| `harl/envs/cascade_reservoir/reward.py` | Ecological rewards and episode accounting |
| `harl/envs/cascade_reservoir/evaluation.py`, `periodic_evaluation.py` | Evaluation and violation counts |
| `harl/envs/cascade_reservoir/checkpoint.py` | Provenance-checked inference replay |
| `analysis/dispatch_diagnostics/` | Ecological dispatch evidence and reports |

Run regression tests with `python -B -m unittest discover -s tests`.
