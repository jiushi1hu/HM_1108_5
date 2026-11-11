"""Unsmoothed seed means and standard deviations for collection and evaluation."""
import argparse
import json
from pathlib import Path
import re
import sys
import numpy as np


VIOLATION_TERMS = ('ecological_flow',)
VIOLATION_SUFFIXES = ('violation_rate', 'violation_count', 'active_reservoir_days')


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(name+' must be a positive integer')
    return int(value)


def evaluation_round(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError('evaluation_round must be a nonnegative integer')
    return int(value)


def window_signature(windows):
    """Identify the fixed scenarios, including initial conditions and forcing."""
    if not isinstance(windows, list) or len(windows) != 3:
        raise ValueError('Periodic evaluation must use exactly three fixed windows')
    fields = ('window_id', 'window_start_year', 'source_years', 'episode_horizon_days',
              'start_date', 'end_date', 'initial_storage_m3', 'initial_level_m',
              'external_inflow_sha256')
    identities = [{key:row[key] for key in fields} for row in windows]
    if len({row['window_id'] for row in identities}) != 3:
        raise ValueError('Periodic window identifiers must be unique')
    for row in identities:
        positive_integer(row['episode_horizon_days'], 'periodic episode_horizon_days')
        if not row['source_years'] or len(set(row['source_years'])) != len(row['source_years']):
            raise ValueError('Periodic source years must be nonempty and unique')
        for name in ('initial_storage_m3', 'initial_level_m'):
            values = np.asarray(row[name], dtype=float)
            if values.shape != (5,) or not np.isfinite(values).all():
                raise ValueError(name+' must contain five finite values')
    return json.dumps(identities, sort_keys=True, allow_nan=False)


def validate_periodic_row(row, expected_signature=None):
    """Keep failures as failures, and never substitute a survivor-only mean."""
    evaluation_round(row['evaluation_round'])
    signature = window_signature(row['windows'])
    if expected_signature is not None and signature != expected_signature:
        raise ValueError('Periodic evaluation scenarios or initial conditions changed')
    if row.get('window_count') != 3:
        raise ValueError('Periodic window_count must equal three')
    statuses = [window.get('status') for window in row['windows']]
    if any(status not in ('ok', 'failed') for status in statuses):
        raise ValueError('Unknown periodic window status')
    successful = statuses.count('ok')
    if row.get('successful_window_count') != successful:
        raise ValueError('Periodic successful_window_count disagrees with the window results')
    values = []
    for window in row['windows']:
        value = window.get('evaluator_return')
        if window['status'] == 'ok':
            if value is None or isinstance(value, bool) or not np.isfinite(float(value)):
                raise ValueError('Successful periodic windows require finite evaluator returns')
            values.append(float(value))
        elif value is not None:
            raise ValueError('Failed periodic windows must not report a completed return')
    value = row.get('evaluator_return')
    if successful == 3:
        if row.get('status') != 'ok' or value is None or isinstance(value, bool):
            raise ValueError('A successful periodic evaluation requires a completed mean return')
        if not np.isfinite(float(value)) or not np.isclose(float(value), np.mean(values), rtol=1e-10, atol=1e-10):
            raise ValueError('Evaluator return must equal the mean of all three fixed windows')
    elif row.get('status') != 'failed' or value is not None:
        raise ValueError('A failed periodic evaluation must leave the aggregate return null')
    validate_violation_metrics(row)
    return signature


def validate_violation_metrics(row):
    """Validate actual violation frequencies, keeping legacy records readable.

    Weighted shortfall penalties measure severity, so they cannot reconstruct
    the number of reservoir-days on which a requirement was violated.
    """
    fields = tuple(term+'_'+suffix for term in VIOLATION_TERMS for suffix in VIOLATION_SUFFIXES)
    scopes = [row]+row['windows']
    if not any(field in scope for scope in scopes for field in fields):
        return False
    if any(field not in scope for scope in scopes for field in fields):
        raise ValueError('Incomplete periodic violation metrics: every window and round require rates, counts and active reservoir-days')
    if row.get('violation_rate_units', 'fraction') != 'fraction':
        raise ValueError('Stored periodic violation rates must be fractions, not percentages')
    for scope in scopes:
        for term in VIOLATION_TERMS:
            rate, count, active = [scope[term+'_'+suffix] for suffix in VIOLATION_SUFFIXES]
            if scope['status'] == 'failed':
                if any(value is not None for value in (rate, count, active)):
                    raise ValueError('Failed periodic evaluations must leave completed violation metrics null')
                continue
            if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0
                   for value in (count, active)) or count > active:
                raise ValueError('Violation counts and active reservoir-days must be nonnegative integers with count <= active')
            if scope is not row and active > 5*scope['episode_horizon_days']:
                raise ValueError('Active reservoir-days exceed the five-reservoir episode horizon')
            if active == 0:
                if rate is not None:
                    raise ValueError('A requirement without active reservoir-days has no defined violation rate')
            elif (rate is None or isinstance(rate, bool) or not np.isfinite(float(rate)) or
                  not 0. <= float(rate) <= 1. or
                  not np.isclose(float(rate), count/active, rtol=1e-10, atol=1e-12)):
                raise ValueError('Violation rate must be the fraction of violating active reservoir-days')
            if scope is not row and 'service_shortfalls' in scope:
                summary = scope['service_shortfalls'][term]
                if summary['reservoir_day_count'] != count or summary['active_reservoir_days'] != active:
                    raise ValueError('Violation counts disagree with the service shortfall summary')
    if row['status'] == 'ok':
        for term in VIOLATION_TERMS:
            for suffix in ('violation_count', 'active_reservoir_days'):
                field = term+'_'+suffix
                if row[field] != sum(window[field] for window in row['windows']):
                    raise ValueError('Aggregate violation counts must sum all three fixed windows')
    return True


def violation_metrics_available(payload):
    """Do not turn absent legacy frequencies into zero-valued curves."""
    _, histories = available_histories(payload, 'evaluation_histories')
    available = [validate_violation_metrics(row) for rows in histories for row in rows]
    if any(available) and not all(available):
        raise ValueError('Some evaluation rounds lack violation counts; mixed legacy/new histories cannot provide a complete violation-rate curve')
    return bool(available) and all(available)


def load_episode_log(path):
    """Stream summary records, retaining collector and evaluator histories separately."""
    configuration, histories, updates = None, {}, {}
    evaluations, last_rounds, scenarios = {}, {}, None
    # Step writers put record_type first. Other key orders still use the JSON
    # parser below, so existing logs do not depend on object-key order.
    step_prefix = re.compile(r'^\s*\{\s*"record_type"\s*:\s*"(?:training_step|evaluation_step)"\s*[,}]')
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if step_prefix.match(line):
                if not line.endswith('\n'):
                    print('Ignoring an incomplete final log record while reading a running experiment.',
                          file=sys.stderr)
                    break
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                if not line.endswith('\n'):
                    print('Ignoring an incomplete final log record while reading a running experiment.',
                          file=sys.stderr)
                    break
                raise ValueError('Invalid JSON at line {}'.format(line_number)) from exc
            if not isinstance(row, dict):
                raise ValueError('Log records must be JSON objects at line {}'.format(line_number))
            kind = row.get('record_type')
            if kind not in ('run_configuration', 'training_update', 'periodic_evaluation'):
                continue
            try:
                if kind == 'run_configuration':
                    if configuration is not None:
                        raise ValueError('A plot input must contain exactly one run configuration')
                    gamma = float(row['config']['algo']['gamma'])
                    if not np.isfinite(gamma) or not 0 <= gamma <= 1:
                        raise ValueError('Gamma must be finite and in [0, 1]')
                    seeds = row['config']['experiment']['seeds']
                    if (not isinstance(seeds, list) or not seeds or
                            any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds) or
                            len(set(seeds)) != len(seeds)):
                        raise ValueError('Configured seeds must be unique integers')
                    configuration = dict(gamma=gamma, configured_seeds=seeds,
                        episode_horizon_days=positive_integer(row['episode_horizon_days'], 'episode_horizon_days'),
                        parallel_envs=positive_integer(row['parallel_environments'], 'parallel_environments'))
                    if 'updates_per_seed' in row:
                        configuration['updates_per_seed'] = positive_integer(row['updates_per_seed'], 'updates_per_seed')
                    periodic = row.get('periodic_evaluation', {})
                    if 'windows' in periodic:
                        scenarios = window_signature(periodic['windows'])
                    configuration['training_years'] = row.get('training_years')
                    configuration['evaluation_source_years'] = row.get('evaluation_source_years', [])
                    continue
                if configuration is None:
                    raise ValueError('A run_configuration must precede training/evaluation results')
                seed = row['seed']
                if isinstance(seed, bool) or not isinstance(seed, int) or seed not in configuration['configured_seeds']:
                    raise ValueError('Result seed is absent from the run configuration')
                if kind == 'periodic_evaluation':
                    number = evaluation_round(row['evaluation_round'])
                    if seed not in last_rounds and number != 0:
                        raise ValueError('Periodic evaluation history must start at round zero')
                    if seed in last_rounds and number <= last_rounds[seed]:
                        raise ValueError('Duplicate or non-increasing evaluation round for seed {}'.format(seed))
                    if number > configuration.get('updates_per_seed', number):
                        raise ValueError('Evaluation round exceeds the configured update budget')
                    signature = validate_periodic_row(row, scenarios)
                    if scenarios is None:
                        scenarios = signature
                    training_years = configuration['training_years']
                    held_out = set(configuration['evaluation_source_years'])
                    for window in row['windows']:
                        years = set(window['source_years'])
                        if years & held_out or (training_years is not None and not years <= set(training_years)):
                            raise ValueError('Periodic evaluation windows must use training years only')
                    if 'gamma' in row and float(row['gamma']) != configuration['gamma']:
                        raise ValueError('Evaluator gamma differs from the run configuration')
                    evaluations.setdefault(seed, []).append(row)
                    last_rounds[seed] = number
                    continue
                if (row['episode_horizon_days'] != configuration['episode_horizon_days'] or
                        row['parallel_envs'] != configuration['parallel_envs']):
                    raise ValueError('Training update horizon/workers disagree with the run configuration')
                step = positive_integer(row['training_steps'], 'training_steps')
                update = positive_integer(row['update'], 'update')
                history = histories.setdefault(seed, [])
                if history and (step <= history[-1]['training_steps'] or update <= updates[seed]):
                    raise ValueError('Duplicate or non-increasing training update for seed {}'.format(seed))
                reward = float(row['episode_reward_mean'])
                generation = float(row['episode_generation_TWh_mean'])
                if not np.isfinite([reward, generation]).all():
                    raise ValueError('Non-finite training curve metric')
                history.append(dict(training_steps=step, episode_reward=reward,
                                    episode_energy_TWh=generation))
                updates[seed] = update
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError('Invalid plotting input at line {}: {}'.format(line_number, exc)) from exc
    if configuration is None:
        raise ValueError('No run_configuration found in the episode log')
    seeds = [seed for seed in configuration['configured_seeds'] if histories.get(seed) or evaluations.get(seed)]
    if not seeds:
        raise ValueError('No training updates or periodic evaluations have been recorded yet')
    return dict(configuration, seeds=seeds, histories=[histories.get(seed, []) for seed in seeds],
                evaluation_histories=[evaluations.get(seed, []) for seed in seeds])


def available_histories(payload, name):
    histories = payload.get(name, [])
    if not histories:
        return [], []
    seeds = payload['seeds']
    if len(histories) != len(seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('One history per unique seed required')
    selected = [(seed, rows) for seed, rows in zip(seeds, histories) if rows]
    return [seed for seed, _ in selected], [rows for _, rows in selected]


def make_figure(payload):
    import matplotlib.pyplot as plt
    seeds, histories = available_histories(payload, 'histories')
    if not histories:
        raise ValueError('One nonempty parallel history per seed required')
    grids = [np.array([r['training_steps'] for r in rows], dtype=float) for rows in histories]
    if any(not np.isfinite(grid).all() or np.any(grid <= 0) or np.any(np.diff(grid) <= 0)
           for grid in grids):
        raise ValueError('Training steps must increase; curves start after collection')
    # A running multi-seed experiment can have completed only some seeds or
    # updates. Average only common observed steps; individual curves stay hidden.
    common_steps = set(grids[0])
    for grid in grids[1:]:
        common_steps.intersection_update(grid)
    mean_grid = np.array(sorted(common_steps), dtype=float)
    gamma = float(payload['gamma'])
    if not np.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError('Parallel curves require gamma in [0, 1]')
    days = positive_integer(payload['episode_horizon_days'], 'episode_horizon_days')
    metrics = ('episode_reward', 'episode_energy_TWh')
    values = [[np.asarray([r[metric] for r in rows], dtype=float) for rows in histories]
              for metric in metrics]
    if not all(np.isfinite(row).all() for series in values for row in series):
        raise ValueError('Non-finite parallel curve metric')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    prefix = 'Mean collected episode'
    for ax, series, label in zip(axes, values,
            (prefix+' net return (gamma={:g})'.format(gamma), prefix+' generation (TWh)')):
        if len(mean_grid):
            aligned = np.array([row[np.searchsorted(grid, mean_grid)] for grid, row in zip(grids, series)])
            mean = aligned.mean(axis=0)
            configured = payload.get('configured_seeds', payload['seeds'])
            mean_label = ('Seed mean' if len(configured) == len(series) else
                          'Available-seed mean (n={})'.format(len(series)))
            ax.plot(mean_grid, mean, lw=2, color='crimson',
                    marker='o' if len(mean_grid) == 1 else None, label=mean_label)
            if len(series) > 1:
                sd = aligned.std(axis=0, ddof=1)
                ax.fill_between(mean_grid, mean-sd, mean+sd, alpha=.15,
                                color='crimson', label='Seed +/-1 SD')
        horizon = 'Five-year (1,825-day)' if days == 1825 else '{}-day diagnostic'.format(days)
        ax.set(xlabel='Collected daily environment transitions', ylabel=label,
               title=horizon+' stochastic training episodes (in-sample)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    return fig


def periodic_histories(payload):
    """Validate one common set of fixed scenarios and evaluation round grids."""
    seeds, histories = available_histories(payload, 'evaluation_histories')
    if not histories:
        raise ValueError('No periodic evaluator records to plot')
    grids, scenarios = [], None
    for seed, rows in zip(seeds, histories):
        grid = np.array([evaluation_round(row['evaluation_round']) for row in rows], dtype=int)
        if grid[0] != 0 or np.any(np.diff(grid) <= 0):
            raise ValueError('Evaluation rounds must start at zero and increase without duplicates')
        for row in rows:
            if row.get('seed') != seed:
                raise ValueError('Periodic history seed disagrees with its curve')
            scenarios = validate_periodic_row(row, scenarios)
        grids.append(grid)
    return seeds, histories, grids


def plot_periodic_series(ax, payload, seeds, grids, values, missing_label):
    """Draw only the seed mean and SD, preserving unavailable-round annotations."""
    from matplotlib.ticker import MaxNLocator
    # Keep missing/failed evaluations as NaN instead of averaging fewer seeds.
    rounds = np.unique(np.concatenate(grids))
    aligned = np.full((len(seeds), len(rounds)), np.nan)
    failed_rounds = set()
    for i, (grid, series) in enumerate(zip(grids, values)):
        aligned[i, np.searchsorted(rounds, grid)] = series
        failed = ~np.isfinite(series)
        failed_rounds.update(grid[failed].tolist())
    if failed_rounds:
        failed_grid = sorted(failed_rounds)
        ax.scatter(failed_grid, np.full(len(failed_grid), .025), marker='x', color='crimson',
                   transform=ax.get_xaxis_transform(), clip_on=False, label=missing_label)
    valid = np.isfinite(aligned).all(axis=0)
    mean = np.full(len(rounds), np.nan)
    mean[valid] = aligned[:, valid].mean(axis=0)
    configured = payload.get('configured_seeds', payload['seeds'])
    mean_label = ('Seed mean' if len(configured) == len(seeds) else
                  'Available-seed mean (n={})'.format(len(seeds)))
    ax.plot(rounds, mean, lw=2, color='crimson', marker='o', markersize=3,
            label=mean_label)
    if len(seeds) > 1:
        sd = np.full(len(rounds), np.nan)
        sd[valid] = aligned[:, valid].std(axis=0, ddof=1)
        ax.fill_between(rounds, mean-sd, mean+sd, alpha=.15, color='crimson', label='Seed +/-1 SD')
    maximum = positive_integer(payload['updates_per_seed'], 'updates_per_seed') if 'updates_per_seed' in payload else max(1, int(rounds.max()))
    if rounds.max() > maximum:
        raise ValueError('Evaluation round exceeds the configured update budget')
    ax.set(xlabel='Evaluation round', xlim=(0, maximum))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)


def make_evaluator_figure(payload):
    """Plot the mean measured return and seed SD; failures remain gaps."""
    import matplotlib.pyplot as plt
    seeds, histories, grids = periodic_histories(payload)
    values = [np.asarray([np.nan if row['evaluator_return'] is None else
                          float(row['evaluator_return']) for row in rows]) for rows in histories]
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    plot_periodic_series(ax, payload, seeds, grids, values, 'Failed evaluation (no return)')
    ax.set(ylabel='Evaluator return (mean of three fixed training windows)',
           title='Evaluator Return vs. Evaluation Round')
    return fig


def make_violation_figure(payload):
    """Plot undiscounted reservoir-day frequencies, independent of penalty weights."""
    import matplotlib.pyplot as plt
    seeds, histories, grids = periodic_histories(payload)
    if not violation_metrics_available(payload):
        raise ValueError('This log does not contain periodic violation counts; violation rates cannot be reconstructed from penalty values')
    fig, axes = plt.subplots(1, 1, figsize=(9, 5), squeeze=False, constrained_layout=True)
    titles = ('Ecological-flow violation rate',)
    for ax, term, title in zip(axes.ravel(), VIOLATION_TERMS, titles):
        field = term+'_violation_rate'
        values = [np.asarray([np.nan if row[field] is None else 100.*float(row[field])
                              for row in rows]) for rows in histories]
        plot_periodic_series(ax, payload, seeds, grids, values, 'Unavailable rate (failure or inactive requirement)')
        ax.set(ylabel='Violation rate (%)', title=title)
    fig.suptitle('Fixed-scenario evaluation: violation rates')
    return fig


def evaluator_output_path(training_output):
    path = Path(training_output)
    return path.with_name(path.stem+'_evaluator_return'+(path.suffix or '.png'))


def violation_output_path(training_output):
    path = Path(training_output)
    return path.with_name(path.stem+'_violation_rates'+(path.suffix or '.png'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode-log', type=Path, help='Read saved JSONL collector and periodic evaluator summaries')
    parser.add_argument('--output', type=Path, help='Save collector curves and derive evaluator/violation figure paths from this stem')
    parser.add_argument('--evaluator-output', type=Path, help='Explicit output path for the separate evaluator figure')
    parser.add_argument('--violations-output', type=Path, help='Explicit output path for ecological-flow violation rates')
    parser.add_argument('--no-show', action='store_true', help='Save without opening an interactive window')
    args = parser.parse_args()
    if args.no_show and all(path is None for path in (args.output, args.evaluator_output, args.violations_output)):
        parser.error('--no-show requires --output, --evaluator-output or --violations-output')
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg', force=True)
    try:
        payload = load_episode_log(args.episode_log) if args.episode_log else json.load(sys.stdin)
        has_training = any(payload.get('histories', []))
        has_evaluation = any(payload.get('evaluation_histories', []))
        if not (has_training or has_evaluation):
            raise ValueError('No collector or evaluator records to plot')
        has_violations = violation_metrics_available(payload) if has_evaluation else False
        if args.violations_output is not None and not has_violations:
            raise ValueError('This log does not contain periodic violation counts; violation rates cannot be reconstructed from penalty values')
        if has_evaluation and not has_violations:
            print('Periodic violation-rate figure skipped: this older log has no violation counts; penalty values cannot reconstruct violation frequencies.', file=sys.stderr)
        evaluator_output = args.evaluator_output
        if evaluator_output is None and args.output is not None and has_evaluation:
            evaluator_output = evaluator_output_path(args.output)
        if args.evaluator_output is not None and not has_evaluation:
            raise ValueError('The log has no periodic evaluator records')
        violations_output = args.violations_output
        if violations_output is None and args.output is not None and has_violations:
            violations_output = violation_output_path(args.output)
        outputs = [path.resolve() for path in (args.output, evaluator_output, violations_output) if path is not None]
        if len(set(outputs)) != len(outputs):
            raise ValueError('Collector, evaluator and violation figures require different output paths')
        if args.episode_log is not None and args.episode_log.resolve() in outputs:
            raise ValueError('A figure cannot overwrite its source log')
        figures = []
        if has_training:
            figures.append((make_figure(payload), args.output, 'training curves'))
        if has_evaluation:
            figures.append((make_evaluator_figure(payload), evaluator_output, 'evaluator curve'))
        if has_violations:
            figures.append((make_violation_figure(payload), violations_output, 'violation-rate curves'))
        for figure, output, label in figures:
            if output is not None:
                output.parent.mkdir(parents=True, exist_ok=True)
                figure.savefig(str(output), dpi=180)
                print('Saved '+label+' to '+str(output.resolve()))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    import matplotlib.pyplot as plt
    if args.no_show:
        for figure, _, _ in figures:
            plt.close(figure)
    else:
        plt.show()


if __name__ == '__main__':
    main()
