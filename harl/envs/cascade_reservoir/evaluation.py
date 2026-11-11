"""Frozen complete-episode evaluation for the five-reservoir cascade."""
import numpy as np
import torch
from harl.envs.cascade_reservoir.data_loader import IDS
from harl.envs.cascade_reservoir.episodes import (
    DEFAULT_ACTION_BINS, DEFAULT_EPISODE_DAYS, episode_environment)
from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade
from harl.envs.cascade_reservoir.training_metrics import log_step, step_record
from harl.envs.cascade_reservoir.reward import (
    EpisodeRewardMetrics, validate_discount_factor, REWARD_AGGREGATION, reward_metadata)


class DailyLimitMetrics:
    """Check executed next-state storage against the action date's daily limits."""

    def __init__(self, env):
        self.env = env
        self.violations = 0
        self.max_bound_excess = 0.

    def update(self):
        env = self.env
        r = env.last_transition
        lower = r.get('storage_lower_m3', env.data.vmin)
        upper = r.get('storage_upper_m3', env.data.vmax)
        below = np.maximum(0., lower-r['storage_end'])
        above = np.maximum(0., r['storage_end']-upper)
        excess = np.maximum(below, above)
        tolerance = np.maximum(1., 1e-10*np.abs(r['storage_start']))
        self.violations += int((excess > tolerance).sum())
        self.max_bound_excess = max(self.max_bound_excess, float(excess.max()))

    def finish(self):
        env = self.env
        if self.violations:
            raise InfeasibleCascade('Evaluation trajectory violates daily next-state storage limits')
        return dict(daily_level_limits=env.level_limits is not None, action_bins=env.action_bins,
            feasible=True, violation_count=self.violations,
            max_daily_bound_excess_m3=self.max_bound_excess)


def _attach_evaluation_progress(error, env, rewards, energy):
    """Preserve executed-day accounting when a physical evaluation step fails.

    The exception type and normal final-evaluation failure behavior are retained.
    Periodic evaluation can use this diagnostic without treating a partial
    trajectory as a complete episode return.
    """
    completed = env.index-env.start
    horizon = env.stop-env.start
    totals = rewards.summary()
    error.evaluation_progress = dict(completed_steps=int(completed),
        episode_step=int(min(completed+1, horizon)),
        date=env.data.dates[min(env.index, env.stop-1)].isoformat(),
        partial_storage_m3=env.storage.tolist(),
        partial_metrics=dict(evaluator_return=totals['episode_reward'],
            system_evaluator_return=totals['episode_system_reward'],
            global_evaluator_return=totals['episode_global_reward'],
            generation_MWh=float(energy), generation_TWh=float(energy/1e6),
            ecological_flow_penalty=totals['episode_penalty_ecological_flow']))


@torch.no_grad()
def evaluate_policy(actors, data, physical_config, seed, episode, training_steps, *,
                    gamma, source_years, episode_days=DEFAULT_EPISODE_DAYS,
                    deterministic=True, diagnostics=False, step_log=None):
    """Evaluate one complete multi-year sequence without collecting training samples.

    Evaluation years are used only after training. A training-only diagnostic
    may explicitly supply training years; mixed split sequences are rejected.
    Source dates and daily level limits are preserved at representative-year joins.
    """
    gamma = validate_discount_factor(gamma)
    if len(actors) != len(IDS):
        raise ValueError('Evaluation requires one actor per reservoir')
    bins = actors[0].args.get('action_bins', DEFAULT_ACTION_BINS)
    if any(actor.args.get('action_bins', DEFAULT_ACTION_BINS) != bins for actor in actors):
        raise ValueError('Evaluation actors must use one common discrete action grid')
    env = episode_environment(data, physical_config, source_years, episode_days, bins)
    used_years = list(dict.fromkeys(day.year for day in env.data.dates[env.start:env.stop]))
    if set(used_years).issubset(data.evaluation_years):
        phase = 'evaluation'
    elif set(used_years).issubset(data.training_years):
        phase = 'training_evaluation'
    else:
        raise ValueError('An evaluation episode cannot mix training and evaluation years')
    initial = env.storage.copy()
    limit_metrics = DailyLimitMetrics(env)
    reward_metrics = EpisodeRewardMetrics(gamma)
    energy, spill, residual = 0., 0., 0.
    energy_by_reservoir, spill_by_reservoir = np.zeros(5), np.zeros(5)
    action_sum, mapped_sum, active_count = np.zeros(5), np.zeros(5), np.zeros(5)
    head_sum, head_min, head_max = np.zeros(5), np.full(5, np.inf), np.full(5, -np.inf)
    for actor in actors:
        actor.prep_rollout()
    for _ in range(env.stop-env.start):
        obs = env.actor_observations()
        actions = [float(actor.act(obs[i:i+1], deterministic=deterministic).item())
                   for i, actor in enumerate(actors)]
        try:
            env.step(np.asarray(actions))
        except InfeasibleCascade as exc:
            _attach_evaluation_progress(exc, env, reward_metrics, energy)
            raise
        row = env.last_transition
        if step_log is not None:
            log_step(step_record(row, seed=seed, episode=episode,
                episode_step=env.index-env.start, phase=phase,
                training_steps=training_steps), step_log)
        reward_metrics.update(row)
        energy += float(row['energy'].sum())
        spill += float(row['spill'].sum()*env.physics.dt)
        residual = max(residual, float(np.abs(row['residual']).max()))
        limit_metrics.update()
        if diagnostics:
            energy_by_reservoir += row['energy']
            spill_by_reservoir += row['spill']*env.physics.dt
            action_sum += row['normalized_actions']*row['controls']
            mapped_sum += row['normalized_actions']**2*row['controls']
            active_count += row['controls']
            head_sum += row['net_head_m']
            head_min = np.minimum(head_min, row['net_head_m'])
            head_max = np.maximum(head_max, row['net_head_m'])
    result = dict(seed=int(seed), episode=int(episode), training_steps=int(training_steps),
        phase=phase, source_years=used_years, window_start_year=used_years[0],
        episode_horizon_days=env.stop-env.start,
        start_date=env.data.dates[env.start].isoformat(),
        end_date=env.data.dates[env.stop-1].isoformat(),
        source_sequence='ordered source years; storage carries across every boundary; leap days retained',
        energy_MWh=energy, energy_TWh=energy/1e6, spill_volume_m3=spill,
        max_balance_residual_m3=residual, head_model=env.physics.head_model,
        head_data_status=env.physics.head_data_status, **env.observation_metadata())
    for i, name in enumerate(IDS):
        result[name+'_start_m3'] = float(initial[i])
        result[name+'_end_m3'] = float(env.storage[i])
    try:
        result.update(limit_metrics.finish())
    except InfeasibleCascade as exc:
        _attach_evaluation_progress(exc, env, reward_metrics, energy)
        raise
    result.update(reward_metrics.summary(detailed=True), **reward_metadata(data),
        reward_penalties=env.reward_settings, reward_aggregation=REWARD_AGGREGATION)
    if diagnostics:
        result.update(energy_by_reservoir_TWh=(energy_by_reservoir/1e6).tolist(),
            spill_by_reservoir_billion_m3=(spill_by_reservoir/1e9).tolist(),
            end_storage_fraction=((env.storage-data.vmin)/(data.vmax-data.vmin)).tolist(),
            total_storage_depletion_billion_m3=float((initial-env.storage).sum()/1e9),
            mean_u=(action_sum/np.maximum(active_count, 1)).tolist(),
            mean_mapping_fraction=(mapped_sum/np.maximum(active_count, 1)).tolist(),
            active_days=active_count.astype(int).tolist(),
            mean_net_head_m=(head_sum/(env.stop-env.start)).tolist(),
            min_net_head_m=head_min.tolist(), max_net_head_m=head_max.tolist())
    return result
