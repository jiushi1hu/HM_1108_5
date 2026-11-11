"""Frozen replay reconstruction and diagnostic accounting integration checks."""
import copy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.dispatch_diagnostics.analyze_run import DailyDispatchWriter, verify_replay
from analysis.dispatch_diagnostics.report import aggregate_rows
from harl.envs.cascade_reservoir.checkpoint import load_checkpoint, save_checkpoint
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.evaluation import evaluate_policy
from harl.envs.cascade_reservoir.periodic_evaluation import select_periodic_windows
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner


class DispatchWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = yaml.safe_load((ROOT/'harl/configs/algos_cfgs/happo_cascade.yaml').read_text())
        cls.physical = yaml.safe_load((ROOT/'harl/configs/envs_cfgs/cascade_reservoir.yaml').read_text())
        args['train'].update(episode_length=3, n_rollout_threads=1)
        args['model'].update(hidden_sizes=[8, 8], action_bins=7)
        args['experiment'].update(episodes=1, seeds=[1])
        cls.data = CascadeData(ROOT/'data/cascade_reservoir_new', cls.physical, args['experiment'])
        cls.runner = CascadeReservoirRunner(cls.data, cls.physical, args, 1,
            periodic_windows=select_periodic_windows(cls.data, episode_days=3))
        # This fixture tests inference/analysis; no optimizer updates are needed.
        cls.runner.update_count = cls.runner.completed_episodes = 1
        cls.runner.training_steps = 3
        cls.directory = tempfile.TemporaryDirectory()
        path = Path(cls.directory.name)/'policy.pt'
        save_checkpoint(cls.runner, path)
        cls.policy = load_checkpoint(path)

    @classmethod
    def tearDownClass(cls):
        cls.runner.close()
        cls.directory.cleanup()

    def test_baseline_matches_saved_policy_and_aggregates_without_double_counting(self):
        stream = io.StringIO()
        writer = DailyDispatchWriter(stream, self.policy, [1965], 'test_window')
        state = copy.deepcopy(self.policy.critic.state_dict())
        rng = torch.get_rng_state().clone()
        result = self.policy.replay([1965], writer=writer)
        reference = evaluate_policy(self.policy.actors, self.data, self.physical,
            seed=1, episode=1, training_steps=3, gamma=self.policy.gamma,
            source_years=[1965], episode_days=3, diagnostics=True)
        self.assertEqual(verify_replay(result, reference, periodic=False)['status'],
                         'matched_original_evaluation')
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in state.items():
            self.assertTrue(torch.equal(value, self.policy.critic.state_dict()[name]))
        self.assertEqual((writer.completed_days, writer.rows), (3, 15))
        np.testing.assert_array_equal(writer.env.storage, result['final_storage_m3'])
        rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
        report = aggregate_rows(rows, dict(gamma=self.policy.gamma))
        self.assertEqual(report['row_count'], 15)
        self.assertEqual(len(report['accounting']), 1)
        account = report['accounting'][0]
        self.assertEqual(account['executed_days'], 3)
        np.testing.assert_allclose(account['generation_MWh'], result['energy_MWh'], rtol=1e-12)
        self.assertAlmostEqual(account['discounted_legacy_team_reward'], result['episode_reward'], places=12)
        self.assertAlmostEqual(account['discounted_system_reward'], result['episode_system_reward'], places=12)
        self.assertAlmostEqual(account['discounted_global_reward'], result['episode_global_reward'], places=12)
        self.assertEqual({row['term'] for row in report['by_phase_term']}, {'ecological_flow'})
        for row in report['by_phase_term']:
            original = result['service_shortfalls'][row['term']]
            self.assertEqual(row['violation_count'], original['reservoir_day_count'])
            self.assertEqual(row['active_reservoir_days'], original['active_reservoir_days'])
        for row in report['mixed_reward_accounting']:
            i = ['WDD', 'BHT', 'XLD', 'XJB', 'THR'].index(row['reservoir_id'])
            self.assertAlmostEqual(row['discounted_mixed_return'], result['episode_mixed_rewards'][i], places=12)
        bad = dict(reference, energy_MWh=reference['energy_MWh']+1.)
        with self.assertRaises(AssertionError):
            verify_replay(result, bad, periodic=False)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            aggregate_rows(rows+rows[:1], dict(gamma=self.policy.gamma))

    def test_partial_or_wrong_order_replay_is_rejected(self):
        writer = DailyDispatchWriter(io.StringIO(), self.policy, [1965], 'test_window')
        with self.assertRaises(ValueError):
            writer.write(dict(reservoir='BHT', episode_step=1))
        with self.assertRaises(ValueError):
            writer.flush()


if __name__ == '__main__':
    unittest.main(verbosity=2)
