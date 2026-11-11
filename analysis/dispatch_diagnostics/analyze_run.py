"""Replay saved final policies and diagnose same-state ecological-flow attainability.

This analysis lives outside harl/examples so it cannot change the implementation
fingerprint required by existing inference checkpoints. No training occurs.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from harl.envs.cascade_reservoir.checkpoint import load_checkpoint
from harl.envs.cascade_reservoir.data_loader import IDS
from harl.envs.cascade_reservoir.episodes import episode_environment
from analysis.dispatch_diagnostics.feasibility import analyze_transition, CLASSIFICATION_DEFINITIONS


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_run(path):
    configuration = completion = None
    periodic, heldout = {}, {}
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            kind = row.get('record_type')
            if kind == 'run_configuration':
                if configuration is not None:
                    raise ValueError('Run log contains multiple configurations')
                configuration = row
            elif kind == 'periodic_evaluation':
                previous = periodic.get(row['seed'])
                if previous is None or row['evaluation_round'] > previous['evaluation_round']:
                    periodic[row['seed']] = row
            elif kind == 'final_evaluation_sequence':
                heldout[row['seed']] = row
            elif kind == 'run_complete':
                completion = row
    if configuration is None or completion is None:
        raise ValueError('Diagnostics require a completed run with configuration metadata')
    return configuration, completion, periodic, heldout


def verify_replay(summary, reference, *, periodic):
    """Compare frozen replay with the measurements originally saved by training."""
    if periodic:
        pairs = [('energy_MWh', 'generation_MWh'), ('episode_reward', 'evaluator_return'),
                 ('episode_system_reward', 'system_evaluator_return'),
                 ('episode_mixed_rewards', 'mixed_evaluator_returns')]
    else:
        pairs = [(name, name) for name in ('energy_MWh', 'episode_reward',
                 'episode_system_reward', 'episode_mixed_rewards', 'spill_volume_m3')]
    maximum = 0.
    for current, original in pairs:
        left = np.asarray(summary[current], dtype=np.float64)
        right = np.asarray(reference[original], dtype=np.float64)
        if left.shape != right.shape or not np.allclose(left, right, rtol=1e-12, atol=1e-9):
            raise AssertionError('Frozen replay does not reproduce original '+original)
        maximum = max(maximum, float(np.max(np.abs(left-right))))
    for term in ('ecological_flow',):
        for key in ('reservoir_day_count', 'active_reservoir_days', 'by_reservoir_days'):
            if summary['service_shortfalls'][term][key] != reference['service_shortfalls'][term][key]:
                raise AssertionError('Frozen replay service counts differ: '+term+' '+key)
    return dict(status='matched_original_evaluation', maximum_absolute_metric_difference=maximum,
                relative_tolerance=1e-12, absolute_tolerance=1e-9)


class DailyDispatchWriter:
    """Reconstruct each baseline day, verify it, then analyze unexecuted alternatives.

    FrozenCascadePolicy supplies five complete replay records per day, including
    saved values and realized returns. A separate environment executes those same
    commands solely to recover the authoritative full-precision transition.
    Counterfactuals never become the next baseline state.
    """

    def __init__(self, stream, policy, source_years, window_id, *, analyze=analyze_transition):
        self.stream, self.policy = stream, policy
        self.env = episode_environment(policy.data, policy.physical_config, source_years,
                                       policy.episode_days, policy.action_bins)
        self.window_id = window_id
        self.source_years = list(dict.fromkeys(day.year for day in self.env.data.dates))
        self.pending, self.writer = [], None
        self.completed_days = self.rows = 0
        self.analyze = analyze

    def write(self, record):
        expected_reservoir = IDS[len(self.pending)]
        if record['reservoir'] != expected_reservoir or record['episode_step'] != self.completed_days+1:
            raise ValueError('Replay records must have five ordered reservoirs per consecutive day')
        self.pending.append(dict(record))
        if len(self.pending) != len(IDS):
            return
        commands = np.asarray([row['command_action_index'] for row in self.pending])
        self.env.step(commands)
        transition = self.env.last_transition
        comparisons = dict(storage_start_m3='storage_start', storage_end_m3='storage_end',
            inflow_m3s='inflow', release_m3s='release', power_MW='power', net_head_m='net_head_m')
        for field, source in comparisons.items():
            actual = np.asarray([row[field] for row in self.pending])
            if not np.array_equal(actual, transition[source]):
                raise AssertionError('Baseline reconstruction differs from replay: '+field)
        if any(row['date'] != transition['date'] for row in self.pending):
            raise AssertionError('Reconstructed source dates differ')
        diagnostics = self.analyze(self.env, transition)
        if len(diagnostics) != len(IDS):
            raise ValueError('Attainability analysis must return every reservoir')
        for i, (diagnostic, replay) in enumerate(zip(diagnostics, self.pending)):
            if diagnostic['reservoir_id'] != IDS[i]:
                raise ValueError('Attainability results must retain hydraulic order')
            row = dict(diagnostic)
            row.update(seed=int(self.policy.metadata['seed']), window_id=self.window_id,
                phase=replay['phase'], policy_update=int(self.policy.metadata['policy_update']),
                timestep_seconds=float(self.env.physics.dt),
                episode_step=replay['episode_step'], episode_year=self.source_years.index(
                    int(replay['date'][:4]))+1, source_years=';'.join(map(str, self.source_years)),
                storage_fraction_start=float((transition['storage_start'][i]-self.policy.data.vmin[i])/
                                              (self.policy.data.vmax[i]-self.policy.data.vmin[i])),
                storage_fraction_end=float((transition['storage_end'][i]-self.policy.data.vmin[i])/
                                            (self.policy.data.vmax[i]-self.policy.data.vmin[i])))
            for name in ('policy_action_index', 'command_action_index', 'effective_action_index',
                         'policy_action_probability', 'entropy', 'spill_m3s', 'spill_volume_m3',
                         'energy_MWh', 'local_reward', 'global_reward', 'mixed_reward',
                         'system_reward', 'legacy_team_reward', 'value', 'next_value', 'td_residual',
                         'realized_mixed_reward_to_go', 'legacy_team_value',
                         'realized_legacy_team_reward_to_go', 'critic_reward_layout',
                         'value_channel', 'value_basis', 'learning_reward',
                         'realized_learning_reward_to_go', 'replay_gae_target', 'replay_gae_advantage',
                         'shared_team_value', 'next_shared_team_value', 'shared_team_td_residual',
                         'realized_shared_team_reward_to_go'):
                row[name] = replay[name]
            row['discount_weight'] = float(self.policy.gamma**(replay['episode_step']-1))
            row['discounted_mixed_reward'] = row['discount_weight']*row['mixed_reward']
            row['discounted_legacy_team_reward'] = row['discount_weight']*row['legacy_team_reward']
            for key, value in row.items():
                if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                    raise ValueError('Non-finite dispatch diagnostic: '+key)
            if self.writer is None:
                self.writer = csv.DictWriter(self.stream, fieldnames=list(row), lineterminator='\n')
                self.writer.writeheader()
            self.writer.writerow(row)
            self.rows += 1
        self.pending.clear()
        self.completed_days += 1
        if self.completed_days % 365 == 0:
            self.stream.flush()
            print('  seed={} window={} diagnosed_days={}/{}'.format(
                self.policy.metadata['seed'], self.window_id, self.completed_days,
                self.policy.episode_days), flush=True)

    def flush(self):
        if self.pending or self.completed_days != self.policy.episode_days:
            raise ValueError('Diagnostic replay did not finish a complete horizon')
        self.stream.flush()


def analyze_saved_run(run_log, output_dir, seeds=None, report_python=None):
    run_log, output_dir = Path(run_log).resolve(), Path(output_dir).resolve()
    configuration, completion, periodic, heldout = read_run(run_log)
    selected = list(configuration['config']['experiment']['seeds'] if seeds is None else seeds)
    if not selected or len(selected) != len(set(selected)) or any(seed not in periodic for seed in selected):
        raise ValueError('Requested seeds must have distinct recorded final evaluations')
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='in_progress', run_log=str(run_log), run_log_sha256=file_hash(run_log),
        seeds=selected, created_utc=datetime.now(timezone.utc).isoformat(),
        gamma=configuration['config']['algo']['gamma'],
        action_bins=configuration['config']['model']['action_bins'],
        checkpoint_policy='final trained policies; no fitting, selection or retraining',
        counterfactual_scope='same starting state, current inflows and daily limits; '
            'only the selected reservoir command changes, all other original commands held fixed',
        interpretation='Same-state restrictions do not prove global physical inevitability; '
            'same-day alternatives do not prove an improvement in long-horizon return. '
            'Alternatives for different reservoirs/days are evaluated separately, not combined.',
        classification_definitions=CLASSIFICATION_DEFINITIONS,
        analysis_source_hashes={str(path.relative_to(ROOT)):file_hash(path)
            for path in sorted(Path(__file__).resolve().parent.glob('*.py'))},
        scenarios=[], checkpoints=[], torch_version=str(torch.__version__), numpy_version=str(np.__version__))
    snapshot_manifest = ROOT/'source_snapshot.json'
    if snapshot_manifest.exists():
        manifest['reference_source_snapshot'] = json.loads(snapshot_manifest.read_text(encoding='utf-8'))
    manifest_path = output_dir/'manifest.json'
    with manifest_path.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    paths = []
    torch.set_num_threads(configuration['config']['device']['torch_threads'])
    for seed in selected:
        final = periodic[seed]
        if final['status'] != 'ok' or final['evaluation_round'] != completion['updates_per_seed'][
                completion['seeds'].index(seed)]:
            raise ValueError('Expected successful final fixed-scenario evaluation for seed '+str(seed))
        directory = Path(configuration['policy_checkpoints']['directories'][str(seed)])
        checkpoint = directory/'update_{:06d}.pt'.format(final['evaluation_round'])
        policy = load_checkpoint(checkpoint)
        if policy.metadata['seed'] != seed or policy.metadata['policy_update'] != final['evaluation_round']:
            raise ValueError('Checkpoint seed/update does not match the saved evaluation')
        manifest['checkpoints'].append(dict(seed=seed, path=str(checkpoint), sha256=file_hash(checkpoint),
            data_hashes=policy.metadata['data_hashes'], source_hashes=policy.metadata['source_hashes'],
            settings_sha256=policy.metadata['settings_sha256'],
            critic_reward_layout=policy.critic_reward_layout, critic_output_dim=policy.critic_output_dim,
            actor_value_channels=list(policy.actor_value_channels)))
        scenarios = [(window['window_id'], window['source_years'], window, True) for window in final['windows']]
        if seed in heldout:
            scenarios.append(('held_out_sequence', heldout[seed]['source_years'], heldout[seed], False))
        for window_id, years, reference, is_periodic in scenarios:
            path = output_dir/'seed{}_{}_dispatch.csv'.format(seed, window_id)
            print('Replaying seed={} window={} years={} ...'.format(seed, window_id, years), flush=True)
            random_before = torch.get_rng_state().clone()
            with path.open('x', encoding='utf-8', newline='') as stream:
                writer = DailyDispatchWriter(stream, policy, years, window_id)
                summary = policy.replay(years, writer=writer)
            if not torch.equal(random_before, torch.get_rng_state()):
                raise AssertionError('Frozen replay changed the torch RNG')
            verification = verify_replay(summary, reference, periodic=is_periodic)
            manifest['scenarios'].append(dict(seed=seed, window_id=window_id, csv=str(path),
                sha256=file_hash(path), rows=writer.rows, verification=verification, replay_summary=summary))
            paths.append(path)
            print('Verified seed={} window={} rows={} energy={:.6f} TWh'.format(
                seed, window_id, writer.rows, summary['energy_TWh']), flush=True)
    manifest['status'] = 'complete'
    manifest['total_daily_reservoir_rows'] = sum(row['rows'] for row in manifest['scenarios'])
    if report_python is None:
        from analysis.dispatch_diagnostics.report import build_report
        build_report(paths, output_dir, manifest)
    else:
        subprocess.run([str(report_python), '-B', str(Path(__file__).with_name('report.py')),
                        '--output-dir', str(output_dir)],
            input=json.dumps(dict(csv_paths=[str(path) for path in paths], metadata=manifest),
                             allow_nan=False), text=True, check=True)
    with manifest_path.open('w', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    print('Diagnostic report: '+str(output_dir/'report.md'), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-log', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; existing outputs are never overwritten')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--report-python', type=Path,
                        help='Optional separate Python interpreter with Matplotlib for report figures')
    args = parser.parse_args()
    analyze_saved_run(args.run_log, args.output_dir, args.seeds, args.report_python)


if __name__ == '__main__':
    main()
