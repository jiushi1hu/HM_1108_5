"""Independent full-physics checks of ecological dispatch attribution."""
import copy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'tests'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.dispatch_diagnostics.feasibility import analyze_transition, DispatchDiagnosticError
from harl.envs.cascade_reservoir.data_loader import CascadeData, IDS
from harl.envs.cascade_reservoir.episodes import episode_environment
from harl.envs.cascade_reservoir.reservoir_physics import ReservoirPhysics
from harl.envs.cascade_reservoir.reward import OperationReward, reward_settings


def synthetic_dispatch(*, head_model='fixed', bins=200, available_water=30.,
                       external=0., outlet=10000., turbine=10000., action=0,
                       downstream_empty=False, ecology=0., reservoir=2):
    """Actual cascade physics with a specified station's water and head curve."""
    seconds = 86400.
    maximum = np.full(5, seconds*10000.)
    storage = np.full(5, seconds*5000.)
    if head_model == 'storage_sensitivity':
        maximum[reservoir] = seconds*100.
        storage[reservoir] = maximum[reservoir]
    else:
        storage[reservoir] = seconds*available_water
    if downstream_empty:
        storage[reservoir+1] = 0.
    anchors = {name: np.asarray([[0., 0.], [100., maximum[i]]])
               for i, name in enumerate(IDS)}
    data = SimpleNamespace(vmin=np.zeros(5), vmax=maximum,
        release_cap=np.full(5, 10000.), turbine_cap=np.full(5, 10000.),
        power_cap=np.full(5, 1e6), head=np.full(5, 100.), anchors=anchors,
        head_sensitivity=None)
    data.release_cap[reservoir], data.turbine_cap[reservoir] = outlet, turbine
    data.levels = lambda values: np.asarray([
        np.interp(values[i], anchors[name][:, 1], anchors[name][:, 0])
        for i, name in enumerate(IDS)])
    if head_model == 'storage_sensitivity':
        slopes, references = np.zeros(5), np.full(5, 100.)
        slopes[reservoir], references[reservoir] = 1., 10.
        data.head_sensitivity = dict(efficiency=np.full(5, .9),
            reference_net_head_m=references, head_slope_m_per_m=slopes,
            assumed_reference_forebay_level_m=np.zeros(5))
    requirements = SimpleNamespace(ecological_flow_m3s=np.zeros(5))
    requirements.ecological_flow_m3s[reservoir] = ecology
    data.service_requirements = requirements
    config = dict(head_model=head_model, release_mapping='quadratic',
        timestep_seconds=seconds, efficiency=.9, water_density_kg_m3=1000.,
        gravity_m_s2=9.81, reward_penalties=dict(enabled=True, ecological_flow_weight=1.))
    physics = ReservoirPhysics(data, config)
    operation = OperationReward(data, reward_settings(config))
    inflow = np.zeros(5)
    inflow[reservoir] = external
    indices = np.zeros(5, dtype=int)
    indices[reservoir] = action
    if downstream_empty:
        indices[reservoir+1] = bins-1
    transition = physics.transition(storage, inflow, indices/(bins-1), data.vmin, data.vmax)
    transition.update(date='2000-01-01', sampled_actions=indices.copy(),
        normalized_actions=transition['actions'].copy(), action_bins=bins,
        timestep_seconds=seconds, levels_start=data.levels(storage),
        levels_end=data.levels(transition['storage_end']))
    transition['actions'] = np.where(transition['controls'] > 0., indices, 0).astype(int)
    operation.apply(transition)
    env = SimpleNamespace(data=data, physics=physics, operation_reward=operation,
        action_bins=bins, storage=transition['storage_end'].copy(), index=1,
        last_transition=transition)
    return env, transition


def brute_grid(env, row, reservoir):
    actions = row['sampled_actions'].astype(float)/(env.action_bins-1)
    results = []
    for index in range(env.action_bins):
        candidate = actions.copy()
        candidate[reservoir] = index/(env.action_bins-1)
        results.append(env.physics.transition(row['storage_start'], row['external'],
            candidate, row['storage_lower_m3'], row['storage_upper_m3'],
            storage_constraints=row.get('storage_constraints')))
    return results


class DispatchFeasibilityTest(unittest.TestCase):
    def test_historical_all_five_exact_grid_against_full_cascade(self):
        config = yaml.safe_load((ROOT/'harl/configs/envs_cfgs/cascade_reservoir.yaml').read_text())
        for head in ('fixed', 'storage_sensitivity'):
            physical = dict(config, head_model=head, terminal_storage_equal_initial=False)
            data = CascadeData(ROOT/'data/cascade_reservoir_new', physical)
            env = episode_environment(data, physical, [1965], days=4, action_bins=21)
            env.operation_reward.requirements = SimpleNamespace(ecological_flow_m3s=np.full(5, 1e6))
            for day in range(4):
                env.step(np.full(5, 20 if day == 0 else 0))
                row = env.last_transition
                records = analyze_transition(env, row)
                self.assertEqual([record['reservoir_id'] for record in records], list(IDS))
                for i, record in enumerate(records):
                    values = [float(candidate['release'][i]) for candidate in brute_grid(env, row, i)]
                    self.assertEqual(record['ecological_flow_best_grid_action_index'], int(np.argmax(values)))
                    np.testing.assert_allclose(record['ecological_flow_best_grid_actual'], max(values),
                                               rtol=1e-10, atol=1e-7)
                    self.assertEqual(record['ecological_flow_meeting_grid_action_count'], 0)
                    self.assertEqual(record['ecological_flow_classification'], 'same_state_release_limit')
                    self.assertTrue(record['ecological_flow_release_limit_proven'])
                json.dumps(records, allow_nan=False)

    def test_terminal_reachability_is_preserved_by_diagnostic_witnesses(self):
        physical = yaml.safe_load((ROOT/'harl/configs/envs_cfgs/cascade_reservoir.yaml').read_text())
        physical.update(head_model='fixed', terminal_storage_equal_initial=True)
        data = CascadeData(ROOT/'data/cascade_reservoir_new', physical)
        env = episode_environment(data, physical, [1965], days=4, action_bins=11)
        initial = env.storage.copy()
        env.operation_reward.requirements = SimpleNamespace(ecological_flow_m3s=np.full(5, 1e6))
        for day in range(4):
            env.step((np.arange(5)*3+day*7) % env.action_bins)
            row = env.last_transition
            records = analyze_transition(env, row)
            for i, record in enumerate(records):
                expected = brute_grid(env, row, i)
                np.testing.assert_allclose(record['ecological_flow_best_grid_actual'],
                    max(float(candidate['release'][i]) for candidate in expected), rtol=1e-10, atol=1e-7)
                for candidate in expected:
                    prefix = np.r_[0., np.cumsum(candidate['storage_end'])]
                    self.assertTrue(np.all(prefix[:, None]-prefix[None, :] <= row['storage_constraints']+1e-3))
                    if day == 3:
                        np.testing.assert_allclose(candidate['storage_end'], initial, rtol=0., atol=1e-3)
        np.testing.assert_allclose(env.storage, initial, rtol=0., atol=1e-3)

    def test_all_station_grid_quantization_and_forced_release_limits(self):
        for i in range(5):
            with self.subTest(reservoir=IDS[i]):
                env, row = synthetic_dispatch(reservoir=i, head_model='storage_sensitivity',
                                              external=100., bins=3, ecology=150.)
                record = analyze_transition(env, row)[i]
                self.assertEqual(record['ecological_flow_classification'], 'same_day_policy_alternative')
                self.assertEqual(record['ecological_flow_minimum_meeting_action_index'], 2)
                self.assertAlmostEqual(record['ecological_flow_continuous_minimum_meeting_release_m3s'], 150., places=6)
                self.assertAlmostEqual(record['ecological_flow_grid_overshoot_above_continuous_minimum_m3s'], 50.)
                self.assertAlmostEqual(record['ecological_flow_meeting_end_storage_delta_m3'], -100.*86400.)
                env, row = synthetic_dispatch(reservoir=i, available_water=0., ecology=1.)
                record = analyze_transition(env, row)[i]
                self.assertTrue(record['forced_action'])
                self.assertEqual(record['ecological_flow_classification'], 'same_state_release_limit')
                self.assertEqual(record['ecological_flow_continuous_max_actual'], 0.)

    def test_original_downstream_commands_and_observational_state(self):
        env, row = synthetic_dispatch(downstream_empty=True, ecology=10.)
        original, storage = copy.deepcopy(row), env.storage.copy()
        numpy_state, python_state = np.random.get_state(), random.getstate()
        with mock.patch.object(env.physics, 'transition', wraps=env.physics.transition) as replay:
            records = analyze_transition(env, row)
        self.assertGreater(replay.call_count, 0)
        for call in replay.call_args_list:
            np.testing.assert_array_equal(call.args[2][[0,1,3,4]],
                row['sampled_actions'][[0,1,3,4]]/(env.action_bins-1))
        for key in row:
            if isinstance(row[key], np.ndarray):
                np.testing.assert_array_equal(row[key], original[key])
        np.testing.assert_array_equal(env.storage, storage)
        self.assertIs(env.last_transition, row)
        self.assertEqual(env.index, 1)
        self.assertEqual(random.getstate(), python_state)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        self.assertEqual(records[2]['ecological_flow_classification'], 'same_day_policy_alternative')
        with mock.patch.object(env.physics, 'transition', side_effect=RuntimeError('broken counterfactual')):
            with self.assertRaises(DispatchDiagnosticError):
                analyze_transition(env, row)
        invalid = copy.deepcopy(row)
        del invalid['sampled_actions']
        with self.assertRaises(DispatchDiagnosticError):
            analyze_transition(env, invalid)

    def test_downstream_storage_bounds_limit_ecological_release(self):
        env, original = synthetic_dispatch(reservoir=0, available_water=1000., ecology=100.)
        data = env.data
        data.release_cap[1] = 30.
        storage = original['storage_start'].copy()
        storage[1] = data.vmax[1]
        env.physics = ReservoirPhysics(data, env.physics.config)
        row = env.physics.transition(storage, original['external'],
            original['sampled_actions']/(env.action_bins-1), data.vmin, data.vmax)
        row.update(date='2000-01-01', sampled_actions=original['sampled_actions'])
        record = analyze_transition(env, row)[0]
        self.assertEqual(record['ecological_flow_classification'], 'same_state_release_limit')
        self.assertEqual(record['ecological_flow_continuous_max_actual'], 30.)

    def test_inactive_and_satisfied_requirements_need_no_replays(self):
        for ecology, action, classification in ((0., 0, 'requirement_inactive'),
                (10., 199, 'no_violation')):
            env, row = synthetic_dispatch(ecology=ecology, action=action)
            with mock.patch.object(env.physics, 'transition', side_effect=AssertionError('unexpected replay')):
                record = analyze_transition(env, row)[2]
            self.assertEqual(record['ecological_flow_classification'], classification)
            self.assertEqual(record['ecological_flow_shortfall'], 0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
