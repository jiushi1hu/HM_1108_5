"""Aggregate frozen-policy dispatch evidence without importing training code.

Aggregation uses the standard library. Matplotlib is imported only for the
three exported figures. Input rows may be dictionaries or paths to daily CSVs.
"""
from collections import Counter, defaultdict
import csv
from datetime import date
import hashlib
import itertools
import json
import math
from pathlib import Path
import textwrap


TERMS = ('ecological_flow',)
RESERVOIRS = ('WDD', 'BHT', 'XLD', 'XJB', 'THR')
_ALIASES = {
    'reservoir_id': ('reservoir_id', 'reservoir'),
    'power_mw': ('power_mw', 'power_MW'),
    'energy_mwh': ('energy_mwh', 'energy_MWh', 'generation_MWh'),
    'ecological_flow_shortfall': ('ecological_flow_shortfall', 'ecological_flow_shortfall_m3s'),
}
_CONTEXT = ('seed', 'window_id', 'phase', 'reservoir_id', 'term')
_PHASE_LABELS = {'training_evaluation': 'Fixed training scenarios',
                 'evaluation': 'Held-out representative-year sequence'}
_UNITS = {'ecological_flow': 'm³/s'}


def _value(row, name, default=None):
    for key in _ALIASES.get(name, (name,)):
        value = row.get(key)
        if value is not None and value != '':
            return value
    return default


def _number(value, name, optional=False):
    if value is None or value == '':
        if optional:
            return None
        raise ValueError('Missing numeric field: '+name)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid numeric field: '+name) from exc
    if not math.isfinite(result):
        raise ValueError('Non-finite numeric field: '+name)
    return result


def _flag(value, name, default=None):
    if value is None or value == '':
        return default
    if value in (True, 1, '1', '1.0', 'True', 'true'):
        return True
    if value in (False, 0, '0', '0.0', 'False', 'false'):
        return False
    raise ValueError('Invalid boolean field: '+name)


class _Stats:
    def __init__(self):
        self.count = 0
        self.total = 0.
        self.minimum = self.maximum = None

    def add(self, value):
        if value is None:
            return
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def finish(self):
        return dict(count=self.count, sum=self.total if self.count else None,
                    mean=self.total/self.count if self.count else None,
                    min=self.minimum, max=self.maximum)


class _Aggregate:
    def __init__(self):
        self.rows = self.active = self.violations = self.meeting = 0
        self.shortfall = self.relative = self.integral = 0.
        self.classes = Counter()
        self.proofs = Counter()
        self.stats = defaultdict(_Stats)
        self.class_stats = defaultdict(lambda: defaultdict(_Stats))
        self.downstream_harm = Counter()

    def add(self, observation):
        self.rows += 1
        self.active += int(observation['active'])
        self.violations += int(observation['violating'])
        self.shortfall += observation['shortfall']
        self.relative += observation['relative_shortfall']
        self.integral += observation['shortfall_integral']
        for key, value in observation['context'].items():
            self.stats[key].add(value)
        if not observation['violating']:
            return
        classification = observation['classification']
        self.classes[classification] += 1
        for key, value in observation['context'].items():
            self.class_stats[classification][key].add(value)
        self.class_stats[classification]['shortfall'].add(observation['shortfall'])
        self.class_stats[classification]['relative_shortfall'].add(observation['relative_shortfall'])
        for key in ('minimum_achievable_shortfall', 'minimum_grid_shortfall'):
            self.stats[key].add(observation[key])
        for key, flag in observation['proofs'].items():
            self.proofs[key] += int(flag)
        if observation['meeting_grid_action_count'] is not None and observation['meeting_grid_action_count'] > 0:
            self.meeting += 1
            for key, value in observation['alternative'].items():
                self.stats[key].add(value)
            for term, worse in observation['downstream_harm'].items():
                if worse is not None:
                    self.downstream_harm[term+'_comparisons'] += 1
                    self.downstream_harm[term+'_increases'] += int(worse)

    def finish(self):
        return dict(rows=self.rows, active_reservoir_days=self.active,
            violation_count=self.violations,
            violation_rate=self.violations/self.active if self.active else None,
            mean_relative_shortfall_all_active_days=self.relative/self.active if self.active else None,
            mean_shortfall_all_active_days=self.shortfall/self.active if self.active else None,
            mean_shortfall_violating_days=self.shortfall/self.violations if self.violations else None,
            shortfall_integral=self.integral,
            classification_counts=dict(sorted(self.classes.items())),
            classification_fractions_among_violations={key:value/self.violations
                for key,value in sorted(self.classes.items())} if self.violations else {},
            proof_counts_among_violations=dict(sorted(self.proofs.items())),
            proof_fractions_among_violations={key:value/self.violations
                for key,value in sorted(self.proofs.items())} if self.violations else {},
            same_day_grid_meeting_alternative_count=self.meeting,
            same_day_grid_meeting_fraction_among_violations=self.meeting/self.violations if self.violations else None,
            same_state_continuous_minimum_shortfall_fraction_of_actual=(
                self.stats['minimum_achievable_shortfall'].total/self.shortfall
                if self.shortfall and self.stats['minimum_achievable_shortfall'].count == self.violations else None),
            context_and_alternative_statistics={key:value.finish() for key,value in sorted(self.stats.items())},
            context_statistics_by_classification={label:{key:stat.finish() for key,stat in sorted(group.items())}
                for label,group in sorted(self.class_stats.items())},
            alternative_other_reservoir_violation_changes=dict(sorted(self.downstream_harm.items())))


def _observe(row, term):
    threshold = _number(_value(row, term+'_threshold'), term+'_threshold', optional=True)
    shortfall = _number(_value(row, term+'_shortfall'), term+'_shortfall')
    if shortfall < 0. or (threshold is not None and threshold < 0.):
        raise ValueError('Service thresholds and shortfalls must be nonnegative')
    classification = str(_value(row, term+'_classification', 'unclassified'))
    active = _flag(_value(row, term+'_active'), term+'_active',
                   threshold > 0. if threshold is not None else classification != 'requirement_inactive')
    violating = _flag(_value(row, term+'_violating'), term+'_violating', shortfall > 0.)
    if violating != (shortfall > 0.) or (violating and not active):
        raise ValueError('Violation flag and actual shortfall disagree')
    relative = _number(_value(row, term+'_relative_shortfall'), term+'_relative_shortfall', optional=True)
    if relative is None:
        if threshold is None and active:
            raise ValueError('Severity requires a threshold or explicit relative shortfall')
        relative = shortfall/threshold if threshold else 0.
    if relative < 0. or (not active and (relative != 0. or shortfall != 0.)):
        raise ValueError('Invalid relative service shortfall')
    integral = _number(_value(row, term+'_shortfall_integral'), term+'_shortfall_integral', optional=True)
    if integral is None:
        dt = _number(_value(row, 'timestep_seconds'), 'timestep_seconds')
        if dt <= 0.:
            raise ValueError('timestep_seconds must be positive')
        integral = shortfall*dt
    if integral < 0.:
        raise ValueError('Shortfall integrals must be nonnegative')
    context = {key:_number(_value(row, key), key, optional=True) for key in
               ('net_head_m', 'storage_start_m3', 'storage_end_m3', 'power_mw', 'release_m3s')}
    low = _number(_value(row, 'storage_lower_m3'), 'storage_lower_m3', optional=True)
    high = _number(_value(row, 'storage_upper_m3'), 'storage_upper_m3', optional=True)
    if low is not None and high is not None and context['storage_end_m3'] is not None and high > low:
        context['end_storage_position_within_daily_bounds'] = (context['storage_end_m3']-low)/(high-low)
    proofs = {name:_flag(_value(row, term+'_'+name), term+'_'+name, False)
              for name in ('release_limit_proven',)}
    meeting = _number(_value(row, term+'_meeting_grid_action_count'), 'meeting_grid_action_count', optional=True)
    alternative = {name:_number(_value(row, term+'_'+name), term+'_'+name, optional=True) for name in
                   ('meeting_release_delta_m3s', 'meeting_end_storage_delta_m3',
                    'minimum_meeting_other_end_storage_delta_m3',
                    'minimum_meeting_other_ecological_violations',
                    'grid_overshoot_above_continuous_minimum_m3s')}
    harm = {}
    for other_term, name in (('ecological_flow', 'ecological'),):
        actual = _number(_value(row, term+'_actual_other_'+name+'_violations',
                               _value(row, 'actual_other_'+name+'_violations')),
                         'actual other violations', optional=True)
        changed = alternative['minimum_meeting_other_'+name+'_violations']
        harm[other_term] = changed > actual if changed is not None and actual is not None else None
    return dict(active=active, violating=violating, classification=classification,
                shortfall=shortfall, relative_shortfall=relative,
                shortfall_integral=integral,
                minimum_achievable_shortfall=_number(_value(row,term+'_minimum_achievable_shortfall'),
                    'minimum_achievable_shortfall',optional=True),
                minimum_grid_shortfall=_number(_value(row,term+'_minimum_grid_shortfall'),
                    'minimum_grid_shortfall',optional=True),
                context=context, proofs=proofs, meeting_grid_action_count=meeting,
                alternative=alternative, downstream_harm=harm)


def _iter_input(source, input_files):
    if isinstance(source, (str, Path)):
        source = [source]
    iterator = iter(source)
    first = next(iterator, None)
    if first is None:
        return
    if isinstance(first, dict):
        for row in itertools.chain((first,), iterator):
            if not isinstance(row, dict):
                raise ValueError('All supplied rows must be dictionaries')
            yield row
    else:
        for name in itertools.chain((first,), iterator):
            path = Path(name)
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(block)
            input_files.append(dict(path=str(path.resolve()), sha256=digest.hexdigest()))
            with path.open(encoding='utf-8', newline='') as stream:
                yield from csv.DictReader(stream)


def _records(groups, names):
    return [dict(zip(names, key), **value.finish()) for key,value in
            sorted(groups.items(), key=lambda pair: tuple(map(str, pair[0])))]


def _gamma(metadata):
    value = metadata.get('gamma', metadata.get('config', {}).get('algo', {}).get('gamma'))
    if value is None:
        return None
    result = _number(value, 'gamma')
    if not 0. <= result <= 1.:
        raise ValueError('gamma must be between zero and one')
    return result


def aggregate_rows(rows_or_csv_paths, metadata=None):
    """Return JSON-compatible counts/severity and context groups from one pass."""
    metadata = metadata or {}
    gamma = _gamma(metadata)
    groups = {name:defaultdict(_Aggregate) for name in
              ('by_context', 'by_month', 'by_episode_year', 'by_phase_reservoir', 'by_phase_term')}
    input_files, seen, years = [], set(), defaultdict(dict)
    accounting = defaultdict(lambda: defaultdict(float))
    mixed_accounting = defaultdict(lambda: defaultdict(float))
    rows_seen = 0
    for row in _iter_input(rows_or_csv_paths, input_files):
        seed = str(_value(row, 'seed'))
        window = str(_value(row, 'window_id'))
        phase = str(_value(row, 'phase'))
        reservoir = str(_value(row, 'reservoir_id'))
        if 'None' in (seed, window, phase, reservoir):
            raise ValueError('Rows require seed/window_id/phase/reservoir_id identifiers')
        day = date.fromisoformat(str(_value(row, 'date')))
        step = _number(_value(row, 'episode_step'), 'episode_step')
        if step < 1 or step != int(step):
            raise ValueError('episode_step must be a positive integer')
        step = int(step)
        identity = (seed, window, phase, step, reservoir)
        if identity in seen:
            raise ValueError('Duplicate diagnostic row: '+str(identity))
        seen.add(identity)
        context = (seed, window, phase)
        if day.year not in years[context]:
            years[context][day.year] = len(years[context])+1
        episode_year = _value(row, 'episode_year', years[context][day.year])
        for term in TERMS:
            observation = _observe(row, term)
            key = context+(reservoir, term)
            groups['by_context'][key].add(observation)
            groups['by_month'][key+(day.month,)].add(observation)
            groups['by_episode_year'][key+(int(episode_year),)].add(observation)
            groups['by_phase_reservoir'][(phase, reservoir, term)].add(observation)
            groups['by_phase_term'][(phase, term)].add(observation)
        weight = gamma**(step-1) if gamma is not None else None
        if reservoir == RESERVOIRS[0]:
            account = accounting[context]
            account['executed_days'] += 1
            for key in ('global_reward', 'system_reward', 'legacy_team_reward'):
                value = _number(_value(row, key), key, optional=True)
                if value is not None:
                    account['undiscounted_'+key] += value
                    if weight is not None:
                        account['discounted_'+key] += weight*value
        energy = _number(_value(row, 'energy_mwh'), 'energy_mwh', optional=True)
        if energy is not None:
            accounting[context]['generation_MWh'] += energy
        value = _number(_value(row, 'mixed_reward'), 'mixed_reward', optional=True)
        if value is not None:
            own = mixed_accounting[context+(reservoir,)]
            own['undiscounted_mixed_return'] += value
            if weight is not None:
                own['discounted_mixed_return'] += weight*value
        rows_seen += 1
    if not rows_seen:
        raise ValueError('No diagnostic rows were supplied')
    result = dict(schema='frozen_dispatch_report_v1', row_count=rows_seen,
        input_files=input_files, gamma=gamma, provenance=metadata,
        definitions=dict(violation_rate='violating active reservoir-days / active reservoir-days; undiscounted',
            mean_relative_shortfall_all_active_days='sum(shortfall / effective threshold) / active reservoir-days; undiscounted',
            conditional_severity='mean physical shortfall on violating days only',
            classification_fractions='classification count / actual violation count; no-violation/inactive rows excluded',
            shortfall_integral_units={'ecological_flow':'m3'},
            daily_storage_position='(end storage - action-date lower bound) / (action-date upper - lower); not global active-storage fraction',
            alternatives='only violating states with a meeting discrete action; deltas refer to the minimum-meeting same-day candidate',
            same_state_residual_severity='sum of continuous minimum achievable deficits / sum of actual deficits in saved violating states; '
                'not a global lower bound across alternative earlier trajectories',
            accounting='repeated global/system/legacy scalars counted once on WDD row per day; mixed returns kept separately per reservoir',
            episode_year_fallback='explicit episode_year preferred; otherwise ordered first appearances of source calendar years per scenario'),
        accounting=[dict(zip(('seed', 'window_id', 'phase'), key), **values)
                    for key,values in sorted(accounting.items())],
        mixed_reward_accounting=[dict(zip(('seed', 'window_id', 'phase', 'reservoir_id'), key), **values)
                                 for key,values in sorted(mixed_accounting.items())])
    dimensions = dict(by_context=_CONTEXT, by_month=_CONTEXT+('calendar_month',),
        by_episode_year=_CONTEXT+('episode_year',),
        by_phase_reservoir=('phase','reservoir_id','term'), by_phase_term=('phase','term'))
    for name, names in dimensions.items():
        result[name] = _records(groups[name], names)
    # Reject overflowing aggregates before serializing them as nonstandard JSON.
    json.dumps(result, allow_nan=False, default=_json_default)
    return result


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError('Metadata contains a non-JSON object: '+type(value).__name__)


def _format(value, percent=False, scale=1., digits=2):
    if value is None:
        return '—'
    return ('{:,.%df}' % digits).format(value*(100. if percent else 1.)/scale)+('%' if percent else '')


def _stat(record, name):
    return record['context_and_alternative_statistics'].get(name, {}).get('mean')


def _markdown(result):
    lines = ['# Frozen-policy dispatch diagnostics', '',
        'This report describes executed frozen-policy trajectories and same-day counterfactuals. '
        'It does not retrain policies or establish long-horizon optimality. Fixed training scenarios '
        'and the held-out representative-year sequence are reported separately.', '',
        'Violation frequency counts active reservoir-days. Relative severity averages the deficit '
        'divided by its effective requirement across **all active days**. Conditional physical '
        'severity averages only violating days. Classification percentages use **violations** as '
        'their denominator; they are not percentages of all days.', '',
        'A meeting same-state action establishes same-day reachability under the tested mapping '
        'and fixed other commands. A physical-limit classification applies to that starting state; '
        'it does not prove that earlier policy decisions could not have preserved more water. '
        'Proof flags can overlap and must not be summed as disjoint causes.', '',
        'The alternative release/storage deltas do not price future water value. Positive release '
        'deltas generally draw additional water; negative end-storage deltas indicate lower ending '
        'storage. Other-reservoir violations are reported where measured.', '']
    if result['gamma'] is not None:
        gamma = result['gamma']
        lines += ['Recorded discount factor: **{}**. Returns use episode-start discounting; '
                  'service counts, severity, and generation below are undiscounted.'.format(gamma), '']
    lines += ['Rows analyzed: **{:,}**. Individual mixed-return accounting and repeated '
              'system/legacy accounting remain separate in `summary.json`. Global, system, '
              'and legacy CSV scalars repeat across stations and are counted once per day.', '',
              '## Frequency and severity', '',
              '| Scope | Term | Active days | Violations | Frequency | Relative severity | Conditional deficit |',
              '|---|---|---:|---:|---:|---:|---:|']
    for r in result['by_phase_term']:
        lines.append('| {} | {} | {:,} | {:,} | {} | {} | {} {} |'.format(
            _PHASE_LABELS.get(r['phase'], r['phase']), r['term'].replace('_',' '),
            r['active_reservoir_days'],r['violation_count'],_format(r['violation_rate'],True),
            _format(r['mean_relative_shortfall_all_active_days'],True),
            _format(r['mean_shortfall_violating_days']),_UNITS[r['term']]))
    phases = sorted({r['phase'] for r in result['by_phase_reservoir']})
    for phase in phases:
        lines += ['', '## '+_PHASE_LABELS.get(phase,phase), '',
            '| Reservoir | Term | Frequency | Relative severity | Meeting grid alternative / violations | Classification mix among violations |',
            '|---|---|---:|---:|---:|---|']
        selected = [r for r in result['by_phase_reservoir'] if r['phase'] == phase]
        for r in selected:
            mix = '; '.join('{}: {}'.format(label.replace('_',' '),_format(fraction,True)) for label,fraction in
                sorted(r['classification_fractions_among_violations'].items(),key=lambda pair:-pair[1])) or 'No violations'
            lines.append('| {} | {} | {} | {} | {} | {} |'.format(r['reservoir_id'],r['term'].replace('_',' '),
                _format(r['violation_rate'],True),_format(r['mean_relative_shortfall_all_active_days'],True),
                _format(r['same_day_grid_meeting_fraction_among_violations'],True),mix))
        lines += ['', '### Measured state and alternative costs', '',
            '| Reservoir | Term | Mean head (m) | Mean ending storage (billion m³) | Meeting alternative days | Mean release delta (m³/s) | Mean ending-storage delta (million m³) |',
            '|---|---|---:|---:|---:|---:|---:|']
        for r in selected:
            lines.append('| {} | {} | {} | {} | {:,} | {} | {} |'.format(r['reservoir_id'],r['term'].replace('_',' '),
                _format(_stat(r,'net_head_m')),_format(_stat(r,'storage_end_m3'),scale=1e9),
                r['same_day_grid_meeting_alternative_count'],_format(_stat(r,'meeting_release_delta_m3s')),
                _format(_stat(r,'meeting_end_storage_delta_m3'),scale=1e6)))
    lines += ['', '## Accounting and coverage', '',
        '| Seed | Scenario | Scope | Days | Generation (TWh) | Discounted system return | Discounted legacy return |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in result['accounting']:
        lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(r['seed'],r['window_id'],
            _PHASE_LABELS.get(r['phase'],r['phase']),int(r.get('executed_days',0)),
            _format(r.get('generation_MWh'),scale=1e6),_format(r.get('discounted_system_reward')),
            _format(r.get('discounted_legacy_team_reward'))))
    lines += ['', '## Artifacts and limits', '',
        '- `summary.json`: full per-seed/scenario/station/term aggregates, calendar-month and episode-year breakdowns, '
        'classification-conditioned head/storage statistics, proof flags, measured alternative costs, and provenance.',
        '- `service_frequency_and_severity.png`: frequency and all-active-day relative severity.',
        '- `classification_composition.png`: observed classification fractions among violations.',
        '- `seasonal_violations.png`: actual calendar-month violation frequencies.',
        '', 'All daily rows are correlated observations. Same-day counterfactuals do not establish '
        'causation, optimality, or the attainability of an alternative starting state. '
        'Missing measurements remain null rather than becoming zeros.', '']
    definitions = result['provenance'].get('classification_definitions', {})
    if isinstance(definitions, dict) and definitions:
        lines += ['## Classification definitions', '']
        lines.extend('- **{}**: {}'.format(label.replace('_',' '),definition)
                     for label,definition in definitions.items())
        lines.append('')
    return '\n'.join(lines).replace('Rows analyzed: **{:,}**.', 'Rows analyzed: **{:,}**.'.format(result['row_count']))


def _plots(result, output):
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    phases = sorted({r['phase'] for r in result['by_phase_reservoir']})
    reservoirs = [r for r in RESERVOIRS if any(x['reservoir_id'] == r for x in result['by_phase_reservoir'])]
    lookup = {(r['phase'],r['reservoir_id'],r['term']):r for r in result['by_phase_reservoir']}
    fig, axes = plt.subplots(len(phases),len(TERMS),figsize=(9,4.5*len(phases)),squeeze=False,constrained_layout=True)
    for j,phase in enumerate(phases):
        for k,term in enumerate(TERMS):
            ax = axes[j,k]
            for shift,key,label,color in ((-.18,'violation_rate','Violating active days','#1e658e'),
                (.18,'mean_relative_shortfall_all_active_days','Mean relative deficit, all active days','#d18726')):
                values = [lookup[phase,r,term][key] for r in reservoirs]
                ax.bar([i+shift for i in range(len(reservoirs))],
                       [float('nan') if v is None else 100*v for v in values],width=.36,label=label,color=color)
            ax.set(xticks=list(range(len(reservoirs))),xticklabels=reservoirs,ylabel='Percent',
                   title=_PHASE_LABELS.get(phase,phase)+'\n'+term.replace('_',' '))
            ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Actual service frequency and severity — distinct quantities')
    fig.savefig(output/'service_frequency_and_severity.png',dpi=170);plt.close(fig)

    labels = sorted({label for r in result['by_phase_reservoir'] for label in r['classification_counts']})
    colors = {label:plt.get_cmap('tab20')(i/max(len(labels)-1,1)) for i,label in enumerate(labels)}
    fig, axes = plt.subplots(len(phases),len(TERMS),figsize=(9,5*len(phases)+1.5),squeeze=False)
    handles = {}
    for j,phase in enumerate(phases):
        for k,term in enumerate(TERMS):
            ax = axes[j,k];bottom = [0.]*len(reservoirs)
            for label in labels:
                values = [100*lookup[phase,r,term]['classification_fractions_among_violations'].get(label,0.) for r in reservoirs]
                bars = ax.bar(reservoirs,values,bottom=bottom,color=colors[label],label=label.replace('_',' '))
                bottom = [a+b for a,b in zip(bottom,values)]
                handles[label] = bars[0]
            for x,r in enumerate(reservoirs):
                ax.text(x,102,'n={:,}'.format(lookup[phase,r,term]['violation_count']),ha='center',fontsize=8)
            ax.set(ylim=(0,112),ylabel='Percent of actual violations',
                   title=_PHASE_LABELS.get(phase,phase)+'\n'+term.replace('_',' '))
            ax.grid(axis='y',alpha=.15)
    if labels:
        fig.legend([handles[label] for label in labels],
                   [textwrap.fill(label.replace('_',' '),38) for label in labels],
                   loc='lower center',ncol=min(3,len(labels)),fontsize=8,frameon=False)
    fig.suptitle('Same-state diagnostic classifications (conditional on a service violation)')
    fig.subplots_adjust(left=.06,right=.98,top=.91,bottom=.17,hspace=.38,wspace=.18)
    fig.savefig(output/'classification_composition.png',dpi=170);plt.close(fig)

    monthly = defaultdict(lambda: [0,0])
    for r in result['by_month']:
        key = (r['phase'],r['reservoir_id'],r['term'],r['calendar_month'])
        monthly[key][0] += r['violation_count'];monthly[key][1] += r['active_reservoir_days']
    fig, axes = plt.subplots(len(phases),len(TERMS),figsize=(9,4.5*len(phases)),squeeze=False,constrained_layout=True)
    for j,phase in enumerate(phases):
        for k,term in enumerate(TERMS):
            ax = axes[j,k]
            for reservoir in reservoirs:
                values = [monthly[phase,reservoir,term,m] for m in range(1,13)]
                ax.plot(range(1,13),[100*a/b if b else float('nan') for a,b in values],marker='o',ms=3,label=reservoir)
            ax.set(xticks=list(range(1,13)),xlabel='Source calendar month',ylabel='Violating active days (%)',
                   title=_PHASE_LABELS.get(phase,phase)+'\n'+term.replace('_',' '))
            ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=5)
    fig.suptitle('Seasonal frequencies pooled across recorded seeds and scenarios')
    fig.savefig(output/'seasonal_violations.png',dpi=170);plt.close(fig)


def build_report(rows_or_csv_paths, output_dir, provenance_or_metadata=None):
    """Write summary.json, readable report.md, and three figures; return paths.

    Output files are exclusive. Existing reports are never overwritten. The
    metadata should include gamma, scenario/split identities, implementation
    and checkpoint hashes, and the diagnostic classification definitions.
    """
    output = Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    names = ('summary.json','report.md','service_frequency_and_severity.png',
             'classification_composition.png','seasonal_violations.png')
    for name in names:
        if (output/name).exists():
            raise FileExistsError('Report output already exists: '+str(output/name))
    result = aggregate_rows(rows_or_csv_paths,provenance_or_metadata)
    _plots(result,output)
    with (output/'summary.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False,default=_json_default)
        stream.write('\n')
    with (output/'report.md').open('x',encoding='utf-8') as stream:
        stream.write(_markdown(result))
    return {name:str((output/name).resolve()) for name in names}


def main():
    """Render CSVs with an interpreter that provides Matplotlib."""
    import argparse
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        request = json.load(sys.stdin)
        if not isinstance(request, dict) or not isinstance(request.get('csv_paths'), list):
            raise ValueError('Standard input must contain an object with csv_paths and metadata')
        metadata = request.get('metadata', {})
        if not isinstance(metadata, dict):
            raise ValueError('metadata must be a JSON object')
        paths = build_report(request['csv_paths'], args.output_dir, metadata)
    except (OSError, ValueError, TypeError, ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps(paths, allow_nan=False))


if __name__ == '__main__':
    main()
