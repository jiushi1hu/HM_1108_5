"""Verify real experiment outputs retain the model-assisted credit used by PPO."""
from contextlib import redirect_stdout
import csv
import gzip
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from examples.train_cascade import load_configuration
from examples import train_cascade_parallel as parallel
from examples.plot_cascade_compare import validate_comparison
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.periodic_evaluation import select_periodic_windows
from harl.envs.cascade_reservoir.checkpoint import load_checkpoint


class CounterfactualPersistenceTest(unittest.TestCase):
    def test_real_experiment_trace_diagnostics_checkpoint_and_plot_identity(self):
        config, physical = load_configuration()
        config['train'].update(episode_length=2,n_rollout_threads=2)
        config['model'].update(hidden_sizes=[8,8],action_bins=7)
        config['algo'].update(ppo_epoch=1,critic_epoch=1,happo_credit_assignment='counterfactual',
                              counterfactual_batch_size=13)
        config['experiment'].update(episodes=2,seeds=[1],diagnostics=True,trajectory_diagnostics=True,
                                    save_checkpoints=True,checkpoint_interval=1)
        data = CascadeData(ROOT/'data/cascade_reservoir_new',physical,config['experiment'])
        windows = select_periodic_windows(data,episode_days=2)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            log = Path(directory)/'experiment.jsonl'
            options = type('Options',(),dict(episode_log=log,no_plot=True,
                skip_final_evaluation=False,check_data_only=False,plot_python=sys.executable))()
            with mock.patch.object(parallel,'PERIODIC_EVALUATION_DAYS',2), \
                    mock.patch.object(parallel,'select_periodic_windows',return_value=windows):
                payload = parallel.run_parallel_experiment(data,physical,config,options)
            validate_comparison(dict(schema='cascade_comparison_v1',runs=[dict(algorithm='happo',payload=payload)]))
            records = [json.loads(line) for line in log.read_text().splitlines()]
            metadata = records[0]
            self.assertEqual(metadata['learning_objective']['advantage_source'],'counterfactual_shared_team')
            self.assertEqual(metadata['learning_objective']['critic_reward_layout'],'shared_team')
            credit_record = next(r for r in records if r['record_type']=='counterfactual_credit')
            self.assertEqual(credit_record['counterfactual_candidates'],2*2*5*7)
            trace = parallel.counterfactual_credit_paths(log,[1],True)[1]
            with gzip.open(trace,'rt',newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows),2*2*5)
            for row in rows:
                probability = np.asarray(json.loads(row['old_policy_probabilities']))
                probability /= probability.sum()
                values = np.asarray(json.loads(row['frozen_q_values']))
                selected = int(row['sampled_action_index'])
                self.assertAlmostEqual(float(row['q_selected']),values[selected],places=8)
                self.assertAlmostEqual(float(row['counterfactual_baseline']),float(probability.dot(values)),places=7)
                self.assertAlmostEqual(float(row['counterfactual_raw_advantage']),
                    float(row['q_selected'])-float(row['counterfactual_baseline']),places=10)
                np.testing.assert_allclose(float(row['counterfactual_training_advantage']),
                    float(row['counterfactual_raw_advantage'])/float(row['advantage_scale']),rtol=1e-6,atol=1e-7)
            trajectory = parallel.trajectory_diagnostics_paths(log,[1],True)[1]
            with trajectory.open(newline='') as stream:
                linked = list(csv.DictReader(stream))
            self.assertEqual(len(linked),len(rows))
            keys = ('seed','update','episode','parallel_env_index','episode_step','reservoir_id')
            original = {tuple(row[k] for k in keys):row for row in rows}
            for row in linked:
                frozen = original[tuple(row[k] for k in keys)]
                self.assertEqual(row['actor_advantage_estimator'],'one_step_counterfactual')
                self.assertEqual(float(row['raw_advantage']),float(frozen['counterfactual_raw_advantage']))
                self.assertEqual(float(row['normalized_advantage']),float(frozen['counterfactual_training_advantage']))
            complete = next(r for r in records if r['record_type']=='run_complete')
            self.assertEqual(len(complete['evaluations']),1)
            checkpoint = load_checkpoint(complete['policy_checkpoints'][0]['path'])
            self.assertEqual(checkpoint.metadata['happo_credit_assignment'],'counterfactual')
            self.assertEqual(checkpoint.metadata['actor_advantage_estimator'],'one_step_counterfactual')
            replay = checkpoint.replay([1965])
            self.assertTrue(np.isfinite(replay['episode_system_reward']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
