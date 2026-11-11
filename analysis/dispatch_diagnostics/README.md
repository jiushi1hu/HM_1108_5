# Ecological-flow dispatch investigation

This analysis replays saved final policies without training or changing the
baseline trajectory. It reconstructs full-precision physical transitions and
examines same-state alternatives for ecological outlet flow.

```bash
python -B analysis/dispatch_diagnostics/analyze_run.py \
  --run-log runs/EXPERIMENT.jsonl --output-dir runs/NEW_dispatch_analysis
```

The output directory must be new. Add `--report-python PATH` to use a separate
interpreter with Matplotlib. Checkpoint source, configuration and data hashes
must match; historical checkpoints require their matching source snapshot.
No integrity check is relaxed by this analysis.

Final fixed training scenarios and any final held-out sequence are replayed for
each selected seed. Replayed energy, returns and ecological violation counts
must reproduce the original run log. For each violated reservoir, all configured
commands are tested with other commands held fixed. Continuous release bounds
identify the minimum release meeting the threshold and the maximum attainable
release. Full-cascade witnesses preserve daily envelopes and hard terminal
reachability. Generation, power and head remain ordinary physical metadata.

## Outputs

- Per-scenario CSVs with five reservoir records per day, physical states,
  commands, frozen values, returns, ecological classifications and witnesses.
- `manifest.json` with source/data/checkpoint fingerprints and replay checks.
- `summary.json` with ecological counts, severity, calendar-month and episode-year
  aggregates, measured alternative deltas and separate reward accounting.
- `report.md` and three PNGs: ecological frequency/severity, classification
  composition and seasonal violation frequency.

## Interpretation

`same_day_policy_alternative` means another configured command meets the
threshold from the saved starting state. `same_state_release_limit` means the
maximum physically admissible release cannot meet it. Inactive thresholds and
nonviolating days are counted separately. A quadratic grid can require more
release than the continuous minimum; this quantization overshoot is recorded.
Because the grid includes both feasible endpoints, it cannot miss an attainable
one-sided ecological threshold entirely.

Alternatives for separate reservoirs/days are evaluated independently and cannot
be combined into a cumulative intervention. An improved same-day ecological
outcome does not establish a better long-horizon return. A release limit in the
reached state does not prove earlier decisions could not have preserved water.

Ecological violation counts, deficit volumes and generation are undiscounted.
Reward accounting retains gamma. Global, system and legacy scalar rewards repeat
across five daily CSV records and are counted once per day; mixed rewards remain
reservoir-specific. Saved critic values estimate returns rather than marginal
water prices.
