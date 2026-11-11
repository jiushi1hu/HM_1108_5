"""Frozen trajectory traces, including forced commands and per-channel scaling."""
import copy
import csv
from datetime import date, timedelta
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harl.envs.cascade_reservoir.data_loader import IDS
from harl.envs.cascade_reservoir.learning_signal import learning_signal_report
from harl.envs.cascade_reservoir.trajectory_diagnostics import (
    TRAJECTORY_DIAGNOSTIC_FIELDS, TrajectoryDiagnosticsWriter)
from harl.models.base.distributions import FixedCategorical


def fixture(days=16, gamma=.9, shared=True):
    """Analytic arrays with intentionally different reward/value-channel scales."""
    lanes, bins, channels = 2, 4, 6
    rewards = np.empty((days, lanes, channels), dtype=np.float32)
    prediction = np.empty_like(rewards)
    advantage = np.empty_like(rewards)
    for t in range(days):
        for lane in range(lanes):
            rewards[t, lane] = (np.arange(channels)+1)*(t+lane+1)
            prediction[t, lane] = (np.arange(channels)+1)*1000+t
            advantage[t, lane] = (np.arange(channels)+1)**2*(t-3*lane)
    targets = prediction+advantage
    next_prediction = np.concatenate((prediction[1:], np.full((1, lanes, channels), 999., np.float32)))
    masks = np.ones((days+1, lanes, 1), np.float32)
    masks[-1] = 0.
    mc = np.empty_like(rewards)
    remaining = np.zeros((lanes, channels), dtype=np.float32)
    for t in reversed(range(days)):
        remaining = rewards[t]+gamma*remaining
        mc[t] = remaining
    probabilities = torch.tensor([[.1, .2, .3, .4]]).repeat(days*lanes, 1)
    distributions = [FixedCategorical(probs=probabilities.clone()) for _ in IDS]
    actors = [SimpleNamespace(use_policy_active_masks=True) for _ in IDS]
    buffers = []
    factors = {}
    for i in range(5):
        active = np.ones_like(masks)
        if i == 0:
            active[0, 0] = 0.
        if i == 4:
            active[:-1] = 0.
        actions = np.full((days, lanes, 1), 3., np.float32)
        actions[active[:-1] == 0.] = 0.
        logs = np.full((days, lanes, 1), np.log(.4), np.float32)
        logs[active[:-1] == 0.] = 0.
        buffers.append(SimpleNamespace(active_masks=active, masks=masks.copy(), actions=actions,
            action_log_probs=logs))
        factors[i] = np.full((days, lanes, 1), 1.+i/10, np.float32)
    runner = SimpleNamespace(actor_buffer=buffers, actor=actors,
        critic_buffer=SimpleNamespace(rewards=rewards, masks=masks,
            bad_masks=np.ones_like(masks), gamma=gamma),
        algo_args={'algo': {'normalize_shared_advantages': shared}})
    before = dict(prediction=prediction, next_prediction=next_prediction,
                  gae_targets=targets, mc=mc, distributions=distributions)
    transitions = []
    for t in range(days):
        rows = []
        for lane in range(lanes):
            controls = np.array([b.active_masks[t, lane, 0] for b in buffers])
            sampled = np.full(5, 3, dtype=np.int64)
            vector = np.arange(5, dtype=float)+t+lane
            row = dict(date=((date(2001, 5, 25) if lane == 0 else date(2001, 10, 1))+
                             timedelta(days=t)).isoformat(), action_bins=bins, timestep_seconds=86400.,
                controls=controls, sampled_actions=sampled,
                actions=np.where(controls > 0., sampled, 0),
                normalized_actions=np.where(controls > 0., 1., .5),
                reward=10., generation_reward=12., global_reward=.75, system_reward=1.25)
            for name in ('storage_start', 'storage_end', 'levels_start', 'levels_end',
                         'storage_lower_m3', 'storage_upper_m3', 'next_state_level_lower_m',
                         'next_state_level_upper_m', 'external', 'inflow', 'release', 'turbine',
                         'net_head_m', 'power', 'energy', 'low', 'high'):
                row[name] = vector.copy()
            row['spill'] = np.full(5, float(t % 2))
            row['residual'] = np.zeros(5)
            row['mixed_rewards'] = rewards[t, lane, :5].copy()
            row['local_rewards'] = vector.copy()
            row['local_generation_rewards'] = vector+1.
            row['local_reward_penalties'] = np.ones(5)
            row['local_reward_penalty_components'] = {
                'ecological_flow': np.ones(5)}
            row['service_shortfalls'] = {'ecological_flow': np.zeros(5)}
            row['service_shortfall_integrals'] = {'ecological_flow': np.zeros(5)}
            rows.append(row)
        transitions.append(rows)
    return runner, before, factors, transitions


def capture(writer, transitions, update=1):
    writer.begin_batch(seed=2, update=update, episode_ids=[1, 2],
                       window_starts=[2001, 2001], source_years=[[2001], [2001]])
    for t, rows in enumerate(transitions):
        writer.capture(t, rows)


def team_fixture(days=3, shared=False):
    """One explicit team target, distinct from the physical mixed rewards."""
    runner, before, factors, transitions = fixture(days=days, shared=shared)
    rewards = -np.arange(days*2, dtype=np.float32).reshape(days, 2, 1)-.25
    for key in ('prediction', 'next_prediction', 'gae_targets'):
        before[key] = before[key][..., :1].copy()
    # A forced-row outlier makes per-actor masked normalization observably
    # different from the required once-per-rollout team normalization.
    before['gae_targets'][0, 0, 0] = before['prediction'][0, 0, 0]+10000.
    remaining = np.zeros((2, 1), dtype=np.float32)
    before['mc'] = np.empty_like(rewards)
    for t in reversed(range(days)):
        remaining = rewards[t]+runner.critic_buffer.gamma*remaining
        before['mc'][t] = remaining
        for lane in range(2):
            transitions[t][lane]['system_reward'] = float(rewards[t, lane, 0])
    runner.critic_buffer.rewards = rewards
    runner.critic_buffer.gae_lambda = .95
    runner.critic_buffer.use_gae = True
    runner.critic_buffer.returns = np.concatenate(
        [before['gae_targets'], np.zeros((1, 2, 1), dtype=np.float32)])
    runner.critic_reward_layout = 'shared_team'
    runner.actor_value_channels = (0, 0, 0, 0, 0)
    runner.critic_output_dim = 1
    runner.num_agents = 5
    runner.batch_window_starts = [2001, 2001]
    runner.batch_source_years = [[2001], [2001]]
    runner.batch_episode_starts = np.zeros(2, dtype=int)
    runner.batch_episode_lengths = [days, days]
    return runner, before, factors, transitions


def read_rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


class TrajectoryDiagnosticsTest(unittest.TestCase):
    def test_frozen_values_own_channels_forced_probabilities_and_summaries(self):
        runner, before, factors, transitions = fixture()
        # A current normalizer or critic must never be consulted after training.
        class ForbiddenNormalizer:
            def denormalize(self, *unused):
                raise AssertionError('Current ValueNorm moments cannot reinterpret old values')
        runner.value_normalizer = ForbiddenNormalizer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace.csv'
            with TrajectoryDiagnosticsWriter(path) as writer:
                capture(writer, transitions)
                transitions[0][0]['release'][:] = -999.
                transitions[0][0]['local_reward_penalty_components']['ecological_flow'][:] = 999.
                summary = writer.write_batch(runner, before, factors)
                self.assertEqual(len(writer._transitions), 0)
                self.assertIsNone(writer._batch)
            rows = read_rows(path)
        self.assertEqual(tuple(rows[0]), TRAJECTORY_DIAGNOSTIC_FIELDS)
        self.assertEqual(len(rows), 16*2*5)
        first = rows[0]
        self.assertEqual(float(first['release_m3s']), 0.)
        self.assertEqual(float(first['local_ecological_flow_penalty']), 1.)
        self.assertEqual(first['forced_action'], '1')
        self.assertEqual(first['sampled_action_index'], '3')
        self.assertEqual(first['effective_action_index'], '0')
        self.assertAlmostEqual(float(first['sampled_action_probability']), .4, places=6)
        self.assertAlmostEqual(float(first['sampled_action_log_probability']), np.log(.4), places=6)
        self.assertEqual(float(first['effective_action_probability']), 1.)
        self.assertEqual(float(first['effective_action_log_probability']), 0.)
        self.assertEqual(float(first['policy_objective_weight']), 0.)
        for row in rows:
            t, lane = int(row['episode_step'])-1, int(row['parallel_env_index'])
            i = IDS.index(row['reservoir_id'])
            self.assertEqual(int(row['value_channel']), i)
            self.assertEqual(row['learning_reward_source'], 'mixed_rewards[{}]'.format(i))
            self.assertEqual(row['value_basis'], 'own_mixed_return')
            self.assertEqual(float(row['own_value_frozen']), float(before['prediction'][t,lane,i]))
            expected_td = (runner.critic_buffer.rewards[t,lane,i]+
                .9*before['next_prediction'][t,lane,i]*(t < 15)-before['prediction'][t,lane,i])
            self.assertAlmostEqual(float(row['td_residual']), expected_td, places=5)
            self.assertEqual(float(row['realized_reward_to_go']), float(before['mc'][t,lane,i]))
            self.assertAlmostEqual(float(row['happo_factor']), 1.+i/10, places=6)
            self.assertEqual(float(row['episode_time_weight']), 1.)
            self.assertEqual(float(row['policy_objective_weight']),
                             float(runner.actor_buffer[i].active_masks[t,lane,0]))
            raw = before['gae_targets'][...,i]-before['prediction'][...,i]
            expected_adv = (raw[t,lane]-raw.mean())/(raw.std()+1e-5)
            self.assertAlmostEqual(float(row['normalized_advantage']), expected_adv, places=6)
        self.assertEqual(summary['rows'], 160)
        self.assertEqual(summary['windows']['all_days']['WDD']['rows'], 32)
        self.assertEqual(summary['windows']['first_14_days']['WDD']['rows'], 28)
        self.assertEqual(summary['windows']['october_may']['WDD']['rows'], 23)
        self.assertEqual(summary['windows']['all_days']['THR']['forced_rows'], 32)
        self.assertEqual(summary['windows']['all_days']['THR']['policy_objective_weight_sum'], 0.)
        self.assertEqual(summary['windows']['all_days']['WDD']['policy_objective_weight_sum'], 31.)
        self.assertEqual(summary['windows']['all_days']['BHT']['policy_objective_weight_sum'], 32.)
        self.assertEqual(summary['windows']['all_days']['BHT']['spill_days'], 16)
        json.dumps(summary, allow_nan=False)

    def test_shared_team_uses_one_value_and_one_common_normalization_with_old_flag_off(self):
        runner, before, factors, transitions = team_fixture()
        raw = before['gae_targets']-before['prediction']
        normalized = (raw-raw.mean(axis=(0, 1), keepdims=True))/(
            raw.std(axis=(0, 1), keepdims=True)+1e-5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'team.csv'
            with TrajectoryDiagnosticsWriter(path) as writer:
                capture(writer, transitions)
                summary = writer.write_batch(runner, before, factors)
            rows = read_rows(path)
        self.assertEqual(summary['schema'], 'cascade_trajectory_v2')
        self.assertEqual(summary['critic_reward_layout'], 'shared_team')
        self.assertEqual(summary['critic_fitted_reward_basis'], 'system_reward')
        self.assertFalse(summary['legacy_team_value_available'])
        self.assertEqual(summary['actor_value_channels'], [0]*5)
        for row in rows:
            t, lane = int(row['episode_step'])-1, int(row['parallel_env_index'])
            self.assertEqual(row['learning_reward_source'], 'system_reward')
            self.assertEqual(row['value_basis'], 'shared_team_return')
            self.assertEqual(row['critic_reward_layout'], 'shared_team')
            self.assertEqual(int(row['value_channel']), 0)
            self.assertEqual(float(row['learning_reward']), float(row['system_reward']))
            self.assertEqual(float(row['own_reward_for_learning']), float(row['learning_reward']))
            self.assertNotEqual(float(row['learning_reward']), float(row['mixed_reward']))
            self.assertEqual(float(row['legacy_team_reward']), 10.)
            self.assertEqual(float(row['own_value_frozen']), float(before['prediction'][t, lane, 0]))
            self.assertEqual(float(row['realized_reward_to_go']), float(before['mc'][t, lane, 0]))
            self.assertAlmostEqual(float(row['normalized_advantage']), normalized[t, lane, 0], places=6)
            expected_td = (runner.critic_buffer.rewards[t, lane, 0]+
                .9*before['next_prediction'][t, lane, 0]*(t < 2)-before['prediction'][t, lane, 0])
            self.assertAlmostEqual(float(row['td_residual']), expected_td, places=5)
        report = learning_signal_report(runner, before, before['prediction'])
        self.assertEqual(report['critic_output_layout'], 'shared_team')
        self.assertEqual(report['critic_fitted_reward_basis'], 'system_reward')
        self.assertFalse(report['legacy_team_value_available'])
        self.assertNotIn('episode_reward', report['per_episode'][0])
        for i, actor in enumerate(report['actors']):
            self.assertEqual(actor['value_channel'], 0)
            self.assertEqual(actor['normalization_scope'], 'shared_team_common_rows')
            active = runner.actor_buffer[i].active_masks[:-1] > 0.
            self.assertEqual(actor['training_advantage']['count'], int(active.sum()))
            if active.any():
                self.assertAlmostEqual(actor['training_advantage']['mean'],
                                       float(normalized[active].mean(dtype=np.float64)))

    def test_shared_team_layout_and_captured_reward_mismatches_are_rejected(self):
        for mismatch in ('channels', 'reward'):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as directory:
                runner, before, factors, transitions = team_fixture(days=2)
                if mismatch == 'channels':
                    runner.actor_value_channels = (0, 1, 0, 0, 0)
                else:
                    transitions[0][0]['system_reward'] += 1.
                path = Path(directory)/'invalid.csv'
                with TrajectoryDiagnosticsWriter(path) as writer:
                    capture(writer, transitions)
                    with self.assertRaises(ValueError):
                        writer.write_batch(runner, before, factors)
                    self.assertEqual(writer.rows_written, 0)
                    self.assertEqual(read_rows(path), [])

    def test_team_critic_fit_is_labelled_separately_from_legacy_episode_accounting(self):
        from harl.envs.cascade_reservoir import diagnostics
        runner, before, _, _ = team_fixture()
        runner.completed_episodes, runner.update_count, runner.training_steps = 2, 1, 6
        runner.value_normalizer = None
        runner.last_reward_metrics = dict(episode_reward=-123., episode_penalty=2.,
            episode_generation_reward=10., episode_penalty_ecological_flow=2.)
        runner.last_rollout = dict(episode_energy_TWh=1.)
        runner.critic_buffer.share_obs = np.zeros((4, 2, 38), dtype=np.float32)
        runner.critic = SimpleNamespace(critic=torch.nn.Linear(1, 1),
            get_values=lambda state: torch.from_numpy(before['prediction'].reshape(-1, 1)))
        before['critic_parameters'] = {}
        before['actor_parameters'] = [{} for _ in IDS]
        for actor in runner.actor:
            actor.actor, actor.clip_param = torch.nn.Linear(1, 1), .2
        with mock.patch.object(diagnostics, 'parameter_update', return_value={}), \
                mock.patch.object(diagnostics, 'actor_distribution',
                                  side_effect=before['distributions']):
            report = diagnostics.after_update(runner, before)
        self.assertEqual(report['critic_fitted_reward_basis'], 'system_reward')
        self.assertFalse(report['legacy_team_value_available'])
        self.assertEqual(report['episode_reward_basis'], 'legacy_team_reward')
        self.assertEqual(report['episode_reward'], -123.)
        self.assertEqual(report['discounted_episode_reward'], -123.)
        self.assertAlmostEqual(report['realized_sampled_return_reward_units'],
                               float(before['mc'][0, :, 0].mean()))
        self.assertNotEqual(report['realized_sampled_return_reward_units'], report['episode_reward'])

    def test_active_only_normalization_gamma_zero_and_missing_services(self):
        runner, before, factors, transitions = fixture(days=3, gamma=0., shared=False)
        before['gae_targets'][0,0,0] = before['prediction'][0,0,0]+10000.
        for rows in transitions:
            for row in rows:
                del row['service_shortfalls']
                del row['service_shortfall_integrals']
                del row['generation_reward']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace.csv'
            with TrajectoryDiagnosticsWriter(path) as writer:
                capture(writer, transitions)
                writer.write_batch(runner, before, factors)
            rows = read_rows(path)
        raw = before['gae_targets'][...,0]-before['prediction'][...,0]
        selected = raw[runner.actor_buffer[0].active_masks[:-1,...,0] > 0.]
        for row in rows:
            t, lane = int(row['episode_step'])-1, int(row['parallel_env_index'])
            self.assertEqual(row['ecological_flow_shortfall_m3s'], '')
            self.assertEqual(float(row['legacy_generation_reward']), 10.)
            self.assertEqual(float(row['episode_time_weight']), 1.)
            self.assertEqual(float(row['policy_objective_weight']),
                             0. if row['forced_action'] == '1' else 1.)
            i = IDS.index(row['reservoir_id'])
            self.assertEqual(float(row['td_residual']),
                             float(runner.critic_buffer.rewards[t,lane,i]-before['prediction'][t,lane,i]))
            self.assertEqual(float(row['realized_reward_to_go']),
                             float(runner.critic_buffer.rewards[t,lane,i]))
            if row['reservoir_id'] == 'WDD':
                expected = (raw[t,lane]-selected.mean())/(selected.std()+1e-5)
                np.testing.assert_allclose(float(row['normalized_advantage']), expected, rtol=1e-6)

    def test_writer_is_observational_and_retains_no_cross_batch_history(self):
        runner, before, factors, transitions = fixture(days=2)
        arrays = copy.deepcopy((before['prediction'], before['next_prediction'], before['gae_targets'],
                                before['mc'], runner.critic_buffer.rewards, factors))
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        probabilities = [dist.probs.clone() for dist in before['distributions']]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace.csv'
            with TrajectoryDiagnosticsWriter(path) as writer:
                capture(writer, transitions)
                one = writer.write_batch(runner, before, factors)
                capture(writer, transitions, update=2)
                two = writer.write_batch(runner, before, factors)
                self.assertEqual(one['rows'], 20)
                self.assertEqual(two['total_rows_written'], 40)
                self.assertFalse(writer._transitions)
            with self.assertRaises(FileExistsError):
                with TrajectoryDiagnosticsWriter(path):
                    pass
        for current, old in zip((before['prediction'], before['next_prediction'], before['gae_targets'],
                                 before['mc'], runner.critic_buffer.rewards), arrays[:5]):
            np.testing.assert_array_equal(current, old)
        for i in factors:
            np.testing.assert_array_equal(factors[i], arrays[5][i])
            self.assertTrue(torch.equal(before['distributions'][i].probs, probabilities[i]))
        self.assertEqual(random.getstate(), python_state)
        after = np.random.get_state()
        self.assertEqual(numpy_state[0], after[0])
        np.testing.assert_array_equal(numpy_state[1], after[1])
        self.assertEqual(numpy_state[2:], after[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_invalid_capture_and_nonfinite_learning_are_rejected(self):
        runner, before, factors, transitions = fixture(days=2)
        with tempfile.TemporaryDirectory() as directory:
            with TrajectoryDiagnosticsWriter(Path(directory)/'trace.csv') as writer:
                with self.assertRaises(ValueError):
                    writer.begin_batch(seed=2, update=1, episode_ids=[1, 1],
                        window_starts=[2001, 2001], source_years=[[2001], [2001]])
                writer.begin_batch(seed=2, update=1, episode_ids=[1, 2],
                    window_starts=[2001, 2001], source_years=[[2001], [2001]])
                with self.assertRaises(ValueError):
                    writer.capture(1, transitions[0])
                invalid = copy.deepcopy(transitions[0])
                invalid[0]['power'][0] = np.nan
                with self.assertRaises(ValueError):
                    writer.capture(0, invalid)
                self.assertEqual(len(writer._transitions), 0)
                writer.capture(0, transitions[0])
                writer.capture(1, transitions[1])
                before['prediction'][0,0,0] = np.inf
                with self.assertRaises(ValueError):
                    writer.write_batch(runner, before, factors)
                self.assertEqual(writer.rows_written, 0)
                self.assertEqual(len(read_rows(writer.path)), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
