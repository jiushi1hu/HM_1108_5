"""Observational ecological-release attainability from saved cascade states.

One reservoir's command varies while the starting state, current inflows,
next-state bounds, terminal reachability matrix and other sampled commands
remain fixed. Same-day alternatives do not establish a better multi-day policy
or prove that a saved state was unavoidable.
"""
import numpy as np

from harl.envs.cascade_reservoir.data_loader import IDS


CLASSIFICATION_DEFINITIONS = {
    'requirement_inactive': 'The effective ecological-flow requirement is zero.',
    'no_violation': 'The executed release meets the ecological requirement within reward tolerance.',
    'same_day_policy_alternative': 'Another discrete command meets the ecological requirement from the same saved state; future water value is unassessed.',
    'same_state_release_limit': 'The maximum feasible release cannot meet the ecological requirement from the same saved state.',
}
SCOPE = ('one command changes; same starting storage, current forcing, next-state bounds, '
         'terminal reachability constraints and other originally sampled commands; '
         'no additional forcing or trajectory changes')
LIMITATION = ('Same-state restrictions do not prove exogenous or globally unavoidable failures. '
              'A meeting alternative may change other ecological releases and future storage; '
              'no long-term optimality or improvement claim is made.')


class DispatchDiagnosticError(RuntimeError):
    """Counterfactual verification failed; the actual trajectory did not fail."""


def _vector(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (5,) or not np.isfinite(value).all():
        raise ValueError(name+' must be a finite five-vector')
    return value


class _TransitionAnalysis:
    def __init__(self, env, row):
        self.env, self.row, self.data, self.physics = env, row, env.data, env.physics
        bins = env.action_bins
        if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer)) or bins < 2:
            raise ValueError('action_bins must be an integer >= 2')
        self.bins = int(bins)
        for name in ('storage_start', 'storage_end', 'external', 'inflow', 'release',
                     'power', 'net_head_m', 'turbine', 'spill', 'low', 'high', 'controls'):
            _vector(row[name], name)
        self.lower = _vector(row.get('storage_lower_m3', self.data.vmin), 'storage lower')
        self.upper = _vector(row.get('storage_upper_m3', self.data.vmax), 'storage upper')
        requirements = getattr(getattr(env, 'operation_reward', None), 'requirements', None)
        if requirements is None:
            requirements = getattr(self.data, 'service_requirements', None)
        if requirements is None:
            raise ValueError('Dispatch diagnostics require loaded ecological requirements')
        self.requirement = _vector(requirements.ecological_flow_m3s, 'ecological flow')
        if (self.requirement < 0.).any():
            raise ValueError('Ecological requirements must be nonnegative')
        sampled = row.get('sampled_actions')
        if sampled is None:
            if not np.all(np.asarray(row['controls']) == 1.):
                raise ValueError('Forced counterfactuals require original sampled_actions')
            normalized = _vector(row['normalized_actions'], 'normalized_actions')
            indices = normalized*(self.bins-1)
            if not np.allclose(indices, np.round(indices), rtol=0., atol=1e-8):
                raise ValueError('Counterfactual commands must identify the discrete grid')
            sampled = np.round(indices)
        sampled = _vector(sampled, 'sampled_actions')
        if ((sampled != np.floor(sampled)) | (sampled < 0) | (sampled >= self.bins)).any():
            raise ValueError('Invalid sampled discrete command')
        self.sampled = sampled.astype(int)
        self.actions = self.sampled.astype(np.float64)/(self.bins-1)
        self.grid = np.arange(self.bins, dtype=np.float64)/(self.bins-1)
        self.start_levels = self.physics._forebay_levels(row['storage_start'])
        self.end_levels = self.physics._forebay_levels(row['storage_end'])
        self.witnesses = {}

    def replay(self, i, normalized, expected_release):
        key = (i, float(normalized))
        if key not in self.witnesses:
            actions = self.actions.copy()
            actions[i] = normalized
            self.witnesses[key] = self.physics.transition(
                self.row['storage_start'], self.row['external'], actions,
                self.lower, self.upper, storage_constraints=self.row.get('storage_constraints'))
        witness = self.witnesses[key]
        if not np.isclose(witness['release'][i], expected_release, rtol=1e-10, atol=1e-7):
            raise DispatchDiagnosticError('Release candidate disagrees with full cascade physics')
        return witness

    def other_count(self, i, row):
        active = self.requirement > 0.
        active = active.copy()
        active[i] = False
        return int(((self.requirement-np.asarray(row['release']) >
                     1e-10*np.maximum(self.requirement, 1.)) & active).sum())

    def witness_fields(self, i, prefix, witness):
        delta = witness['storage_end']-self.row['storage_end']
        return {
            prefix+'_other_ecological_violations': self.other_count(i, witness),
            prefix+'_end_storage_delta_m3': float(delta[i]),
            prefix+'_other_end_storage_delta_m3': float(delta.sum()-delta[i]),
        }

    def service(self, i):
        threshold = float(self.requirement[i])
        tolerance = 1e-10*max(threshold, 1.)
        actual = float(self.row['release'][i])
        shortfall = max(threshold-actual, 0.)
        shortfall = shortfall if shortfall > tolerance else 0.
        record = dict(active=bool(threshold > 0.), violating=bool(shortfall),
            threshold=threshold, actual=actual, shortfall=shortfall,
            relative_shortfall=shortfall/threshold if threshold else 0.,
            shortfall_integral=shortfall*self.physics.dt,
            classification='requirement_inactive' if threshold == 0. else 'no_violation',
            release_limit_proven=False)
        optional = ('best_grid_action_index', 'best_grid_actual', 'best_grid_release_m3s',
            'meeting_grid_action_count', 'minimum_meeting_action_index',
            'minimum_meeting_release_m3s', 'meeting_release_delta_m3s',
            'meeting_end_storage_delta_m3', 'continuous_max_actual',
            'continuous_best_release_m3s', 'continuous_minimum_meeting_release_m3s',
            'continuous_minimum_meeting_normalized_action', 'continuous_meeting_release_delta_m3s',
            'minimum_achievable_shortfall', 'minimum_grid_shortfall',
            'grid_overshoot_above_continuous_minimum_m3s')
        record.update(dict.fromkeys(optional))
        for prefix in ('best_grid', 'minimum_meeting', 'continuous_best', 'continuous_minimum_meeting'):
            for suffix in ('other_ecological_violations', 'end_storage_delta_m3',
                           'other_end_storage_delta_m3'):
                record[prefix+'_'+suffix] = None
        if not shortfall:
            return record
        low, high = float(self.row['low'][i]), float(self.row['high'][i])
        if low > high+1e-8 or low < -1e-8:
            raise ValueError('Invalid feasible release interval')
        forced = high-low <= 1e-8
        releases = np.full(self.bins, actual) if forced else low+self.grid**2*(high-low)
        best = int(np.argmax(releases))
        witness = self.replay(i, float(self.grid[best]), float(releases[best]))
        maximum = float(witness['release'][i])
        minimum_shortfall = max(0., threshold-maximum)
        minimum_shortfall = minimum_shortfall if minimum_shortfall > tolerance else 0.
        meeting = np.flatnonzero(threshold-releases <= tolerance)
        record.update(best_grid_action_index=best, best_grid_actual=maximum,
            best_grid_release_m3s=maximum, meeting_grid_action_count=int(len(meeting)),
            continuous_max_actual=maximum, continuous_best_release_m3s=maximum,
            minimum_grid_shortfall=minimum_shortfall,
            minimum_achievable_shortfall=minimum_shortfall)
        record.update(self.witness_fields(i, 'best_grid', witness))
        record.update(self.witness_fields(i, 'continuous_best', witness))
        # Release grows monotonically with command, and the grid contains both
        # endpoints. Any continuously attainable ecological minimum therefore
        # also has a meeting grid command. Quantization changes water use, not
        # attainability of this lower-bound release requirement.
        if not len(meeting):
            record.update(classification='same_state_release_limit', release_limit_proven=True)
            return record
        first = int(meeting[0])
        witness = self.replay(i, float(self.grid[first]), float(releases[first]))
        record.update(classification='same_day_policy_alternative',
            minimum_meeting_action_index=first,
            minimum_meeting_release_m3s=float(witness['release'][i]),
            meeting_release_delta_m3s=float(witness['release'][i]-actual),
            meeting_end_storage_delta_m3=float(witness['storage_end'][i]-self.row['storage_end'][i]))
        record.update(self.witness_fields(i, 'minimum_meeting', witness))
        continuous_release = actual if forced else min(high, max(low, threshold))
        normalized = (.5 if forced else float(np.sqrt(np.clip(
            (continuous_release-low)/(high-low), 0., 1.))))
        witness = self.replay(i, normalized, continuous_release)
        if threshold-float(witness['release'][i]) > tolerance+1e-7:
            raise DispatchDiagnosticError('Continuous meeting witness fails its requirement')
        record.update(continuous_minimum_meeting_release_m3s=float(witness['release'][i]),
            continuous_minimum_meeting_normalized_action=normalized,
            continuous_meeting_release_delta_m3s=float(witness['release'][i]-actual),
            grid_overshoot_above_continuous_minimum_m3s=max(0.,
                record['minimum_meeting_release_m3s']-float(witness['release'][i])))
        record.update(self.witness_fields(i, 'continuous_minimum_meeting', witness))
        return record

    def records(self):
        records = []
        for i, name in enumerate(IDS):
            row = self.row
            record = dict(date=str(row.get('date', 'unknown')), reservoir_id=name, reservoir_index=i,
                action_bins=self.bins, sampled_action_index=int(self.sampled[i]),
                forced_action=bool(row['controls'][i] <= 0.),
                storage_start_m3=float(row['storage_start'][i]), storage_end_m3=float(row['storage_end'][i]),
                level_start_m=float(self.start_levels[i]), level_end_m=float(self.end_levels[i]),
                storage_lower_m3=float(self.lower[i]), storage_upper_m3=float(self.upper[i]),
                inflow_m3s=float(row['inflow'][i]), external_inflow_m3s=float(row['external'][i]),
                release_m3s=float(row['release'][i]), turbine_m3s=float(row['turbine'][i]),
                spill_m3s=float(row['spill'][i]), power_mw=float(row['power'][i]),
                net_head_m=float(row['net_head_m'][i]),
                feasible_release_low_m3s=float(row['low'][i]), feasible_release_high_m3s=float(row['high'][i]),
                outlet_capacity_m3s=float(self.data.release_cap[i]),
                turbine_capacity_m3s=float(self.data.turbine_cap[i]),
                installed_capacity_mw=float(self.data.power_cap[i]),
                actual_other_ecological_violations=self.other_count(i, row))
            record.update({'ecological_flow_'+key:value for key,value in self.service(i).items()})
            record['ecological_flow_shortfall_m3s'] = record['ecological_flow_shortfall']
            records.append(record)
        return records


def analyze_transition(env, transition):
    """Return five ecological diagnostic rows without modifying inputs or RNGs.

    Only violated requirements trigger alternative searches. Grid, maximum
    release and minimum-meeting witnesses execute full cascade physics with
    the saved terminal reachability constraints and other commands fixed.
    Diagnostic errors never classify the actual trajectory as infeasible.
    """
    try:
        return _TransitionAnalysis(env, transition).records()
    except DispatchDiagnosticError:
        raise
    except Exception as exc:
        raise DispatchDiagnosticError('Same-state dispatch diagnostic failed at {}: {}'.format(
            transition.get('date', 'unknown'), exc)) from exc
