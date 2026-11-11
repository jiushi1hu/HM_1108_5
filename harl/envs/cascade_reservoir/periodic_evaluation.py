"""Fixed training-only scenarios for observational HAPPO policy evaluation."""
from contextlib import contextmanager
import hashlib
import random

import numpy as np
import torch

from harl.envs.cascade_reservoir.data_loader import IDS
from harl.envs.cascade_reservoir.episodes import positive_integer, training_windows
from harl.envs.cascade_reservoir.evaluation import evaluate_policy
from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade
from harl.envs.cascade_reservoir.reward import REWARD_AGGREGATION, validate_discount_factor


PERIODIC_EVALUATION_DAYS = 1825
_SCALAR_METRICS = ('evaluator_return', 'generation_MWh', 'generation_TWh',
                   'ecological_flow_penalty',
                   'system_evaluator_return', 'global_evaluator_return')
_VECTOR_METRICS = ('final_storage_m3', 'final_level_m',
                   'mixed_evaluator_returns', 'local_evaluator_returns')
_VIOLATION_TERMS = ('ecological_flow',)
_VIOLATION_METRICS = tuple(term+suffix for term in _VIOLATION_TERMS
    for suffix in ('_violation_rate', '_violation_count', '_active_reservoir_days'))


def _violation_metrics(shortfalls):
    """Report service frequency from undiscounted active reservoir-day counts.

    Relative shortfall magnitudes and weighted reward penalties cannot recover
    these frequencies. A disabled requirement contributes no eligible days;
    a term with no eligible reservoir-days has an undefined (null) rate.
    """
    result = {}
    for term in _VIOLATION_TERMS:
        summary = shortfalls[term]
        count, eligible = summary['reservoir_day_count'], summary['active_reservoir_days']
        if any(isinstance(value, (bool, np.bool_)) or
               not isinstance(value, (int, np.integer)) or value < 0
               for value in (count, eligible)) or count > eligible:
            raise ValueError('Service violation counts must satisfy 0 <= count <= active reservoir-days')
        result.update({term+'_violation_count':int(count),
                       term+'_active_reservoir_days':int(eligible),
                       term+'_violation_rate':float(count/eligible) if eligible else None})
    return result


def _window_descriptor(data, start_year, years, episode_days, label):
    start, _ = data.indices(start_year, start_year)
    stop = start+episode_days
    external = np.asarray(data.external[start:stop], dtype=np.float64)
    initial = data.initial_storage(start_year)
    return dict(window_id=label+'_'+str(start_year), label=label,
        window_start_year=int(start_year), source_years=[int(year) for year in years],
        episode_horizon_days=episode_days, start_date=data.dates[start].isoformat(),
        end_date=data.dates[stop-1].isoformat(), initial_storage_m3=initial.tolist(),
        initial_level_m=data.levels(initial).tolist(),
        mean_total_external_inflow_m3s=float(external.sum(axis=1).mean()),
        external_inflow_sha256=hashlib.sha256(external.tobytes()).hexdigest())


def select_periodic_windows(data, episode_days=PERIODIC_EVALUATION_DAYS):
    """Select dry, median and wet complete training windows without using RNGs.

    Windows are ranked by their mean total exogenous inflow, with start year
    resolving ties. These labels describe the three training-window quantiles,
    not the held-out representative-year classifications. Selection statistics
    are used only to fix evaluation scenarios, never as policy observations.
    The default five-year horizon is independent of collection episode length.
    An explicit shorter horizon supports focused integration checks.
    """
    days = positive_integer(episode_days, 'periodic_evaluation_days')
    candidates = training_windows(data, days)
    if len(candidates) < 3:
        raise ValueError('Periodic evaluation requires at least three complete training-only windows')
    ranked = []
    for year, years in candidates.items():
        start, _ = data.indices(year, year)
        mean = float(data.external[start:start+days].sum(axis=1).mean())
        ranked.append((mean, int(year), years))
    ranked.sort(key=lambda item: (item[0], item[1]))
    selected = (ranked[0], ranked[len(ranked)//2], ranked[-1])
    return [_window_descriptor(data, year, years, days, label)
            for label, (_, year, years) in zip(('dry', 'median', 'wet'), selected)]


def _validated_windows(data, windows):
    if not isinstance(windows, (list, tuple)) or len(windows) != 3:
        raise ValueError('Periodic evaluation requires exactly three fixed windows')
    starts, identities, validated = set(), set(), []
    horizons = set()
    for window in windows:
        if not isinstance(window, dict):
            raise ValueError('Periodic evaluation windows must be descriptor dictionaries')
        days = positive_integer(window['episode_horizon_days'], 'periodic_evaluation_days')
        start = positive_integer(window['window_start_year'], 'window_start_year')
        candidates = training_windows(data, days)
        if start not in candidates or window['source_years'] != candidates[start]:
            raise ValueError('Periodic evaluation window must be a complete consecutive training-only window')
        if set(window['source_years']) & set(data.evaluation_years):
            raise ValueError('Periodic evaluation must not include final held-out evaluation years')
        if start in starts or window['window_id'] in identities:
            raise ValueError('Periodic evaluation windows must have distinct starts and identities')
        expected = _window_descriptor(data, start, candidates[start], days, window['label'])
        if expected != window:
            raise ValueError('Periodic evaluation scenario dates, inflows or initial conditions changed')
        starts.add(start)
        identities.add(window['window_id'])
        horizons.add(days)
        validated.append(dict(window))
    if len(horizons) != 1:
        raise ValueError('All periodic evaluation windows must have the same episode horizon')
    return validated


@contextmanager
def _preserve_training_state(actors):
    """Evaluation consumes no training RNG draws and restores every module mode."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    modes = [(module, module.training) for actor in actors for module in actor.actor.modules()]
    try:
        with torch.no_grad():
            yield
    finally:
        for module, mode in modes:
            module.training = mode
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def evaluate_periodic_policy(actors, data, physical_config, *, seed, evaluation_round,
                             training_steps, windows, gamma):
    """Measure a frozen policy on all three scenarios without changing training.

    Each scenario gets a newly reset environment and deterministic argmax
    actions. Physical failures are logged per scenario, and invalidate the
    entire round mean: surviving windows are never silently reweighted.
    Programming/configuration errors propagate instead of masquerading as
    hydrological infeasibility. No evaluation rows enter training buffers.
    """
    if (isinstance(evaluation_round, (bool, np.bool_)) or
            not isinstance(evaluation_round, (int, np.integer)) or evaluation_round < 0):
        raise ValueError('evaluation_round must be a nonnegative integer')
    if (isinstance(training_steps, (bool, np.bool_)) or
            not isinstance(training_steps, (int, np.integer)) or training_steps < 0):
        raise ValueError('training_steps must be a nonnegative integer')
    gamma = validate_discount_factor(gamma)
    scenarios = _validated_windows(data, windows)
    individual = []
    with _preserve_training_state(actors):
        for window in scenarios:
            result = dict(window, seed=int(seed), evaluation_round=int(evaluation_round),
                          status='ok', failure=None)
            try:
                measured = evaluate_policy(actors, data, physical_config, seed=int(seed),
                    episode=int(evaluation_round), training_steps=int(training_steps),
                    gamma=gamma, source_years=window['source_years'],
                    episode_days=window['episode_horizon_days'], deterministic=True)
            except InfeasibleCascade as exc:
                progress = getattr(exc, 'evaluation_progress', {})
                result.update(status='failed', failure=dict(error_type=type(exc).__name__,
                    reason=str(exc), date=progress.get('date', window['start_date']),
                    episode_step=progress.get('episode_step', 1),
                    completed_steps=progress.get('completed_steps', 0)),
                    partial_metrics=progress.get('partial_metrics', {}),
                    partial_storage_m3=progress.get('partial_storage_m3', window['initial_storage_m3']))
                result.update({metric:None for metric in
                               _SCALAR_METRICS+_VECTOR_METRICS+_VIOLATION_METRICS})
                result['service_shortfalls'] = None
            else:
                final_storage = [measured[name+'_end_m3'] for name in IDS]
                result.update(evaluator_return=measured['episode_reward'],
                    system_evaluator_return=measured['episode_system_reward'],
                    global_evaluator_return=measured['episode_global_reward'],
                    mixed_evaluator_returns=measured['episode_mixed_rewards'],
                    local_evaluator_returns=measured['episode_local_rewards'],
                    generation_MWh=measured['energy_MWh'], generation_TWh=measured['energy_TWh'],
                    ecological_flow_penalty=measured['episode_penalty_ecological_flow'],
                    final_storage_m3=final_storage,
                    final_level_m=data.levels(final_storage).tolist(),
                    completed_steps=window['episode_horizon_days'],
                    violation_count=measured['violation_count'],
                    max_daily_bound_excess_m3=measured['max_daily_bound_excess_m3'],
                    service_shortfalls=measured['service_shortfalls'])
                result.update(_violation_metrics(measured['service_shortfalls']))
            individual.append(result)
    successes = sum(row['status'] == 'ok' for row in individual)
    record = dict(record_type='periodic_evaluation', phase='training_evaluation',
        seed=int(seed), evaluation_round=int(evaluation_round), training_steps=int(training_steps),
        status='ok' if successes == 3 else 'failed', window_count=3,
        successful_window_count=successes, gamma=gamma, deterministic=True,
        reservoir_ids=list(IDS), evaluator_return_aggregation=REWARD_AGGREGATION,
        scenario_aggregation='arithmetic mean over all three fixed windows; null when any window fails',
        violation_rate_definition='violating active reservoir-days / all active reservoir-days; '
            'undiscounted and independent of reward weights and shortfall magnitudes',
        violation_rate_units='fraction',
        violation_rate_aggregation='sum counts / sum active reservoir-days across all three complete windows; '
            'null when any window fails or the denominator is zero',
        windows=individual)
    for metric in _SCALAR_METRICS+_VECTOR_METRICS:
        record[metric] = (np.asarray([row[metric] for row in individual], dtype=np.float64).mean(axis=0).tolist()
                          if successes == 3 else None)
    for term in _VIOLATION_TERMS:
        count_key, eligible_key = term+'_violation_count', term+'_active_reservoir_days'
        count = sum(row[count_key] for row in individual) if successes == 3 else None
        eligible = sum(row[eligible_key] for row in individual) if successes == 3 else None
        record.update({count_key:count, eligible_key:eligible,
                       term+'_violation_rate':count/eligible if eligible else None})
    return record
