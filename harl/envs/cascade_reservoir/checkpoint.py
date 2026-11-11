"""Versioned, provenance-checked inference checkpoints and deterministic replay.

These files intentionally contain no optimizer or sampler state and cannot resume
training. Only load trusted local checkpoints, particularly on older PyTorch
versions which do not support the restricted ``weights_only`` loader.
"""
import copy
import csv
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from harl.common.valuenorm import ValueNorm
from harl.envs.cascade_reservoir.data_loader import CascadeData, IDS
from harl.envs.cascade_reservoir.episodes import episode_environment, positive_integer
from harl.envs.cascade_reservoir.evaluation import DailyLimitMetrics, _attach_evaluation_progress
from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade
from harl.envs.cascade_reservoir.reward import (
    EpisodeRewardMetrics, SYSTEM_REWARD_DEFINITION, reward_metadata, validate_discount_factor)
from harl.models.policy_models.stochastic_policy import StochasticPolicy
from harl.models.value_function_models.v_net import VNet


ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_FORMAT = 'cascade_reservoir_inference'
CHECKPOINT_VERSION = 1
REPLAY_TRACE_FIELDS = (
    'phase', 'seed', 'policy_update', 'episode_step', 'date', 'reservoir',
    'policy_action_index', 'command_action_index', 'effective_action_index',
    'action_overridden', 'controllable', 'normalized_action', 'mapping_fraction',
    'policy_action_probability', 'command_probability', 'command_log_probability', 'entropy',
    'storage_start_m3', 'storage_end_m3', 'level_start_m', 'level_end_m',
    'external_inflow_m3s', 'inflow_m3s', 'release_m3s', 'turbine_flow_m3s', 'spill_m3s',
    'spill_volume_m3', 'net_head_m', 'power_MW', 'energy_MWh',
    'release_lower_m3s', 'release_upper_m3s', 'storage_lower_m3', 'storage_upper_m3',
    'next_state_level_lower_m', 'next_state_level_upper_m', 'balance_residual_m3',
    'local_generation_reward', 'local_reward_penalty', 'local_reward', 'global_reward',
    'mixed_reward', 'system_reward', 'legacy_team_reward',
    'ecological_flow_shortfall_m3s', 'ecological_flow_deficit_m3',
    'critic_reward_layout', 'value_channel', 'value_basis', 'learning_reward',
    'value', 'next_value', 'td_residual', 'realized_learning_reward_to_go',
    'replay_gae_target', 'replay_gae_advantage',
    'shared_team_value', 'next_shared_team_value', 'shared_team_td_residual',
    'realized_shared_team_reward_to_go', 'realized_mixed_reward_to_go',
    'legacy_team_value', 'realized_legacy_team_reward_to_go',
)


def _plain(value):
    """Store metadata as JSON-compatible primitives, never application objects."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError('Unsupported checkpoint metadata: '+type(value).__name__)


def _digest(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_hashes():
    # Include configs as well as their effective overrides. Tests and output
    # artifacts are deliberately outside the implementation provenance boundary.
    paths = list((ROOT/'harl').rglob('*.py')) + list((ROOT/'examples').glob('*.py'))
    paths += list((ROOT/'harl/configs').rglob('*.yaml')) + [ROOT/'setup.py']
    return {str(p.relative_to(ROOT)): _file_hash(p) for p in sorted(paths)}


def _state_copy(module):
    result = {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
    _state_hash(result)
    return result


def _state_hash(state):
    if not isinstance(state, dict) or not state:
        raise ValueError('Checkpoint model state must be a nonempty tensor mapping')
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError('Checkpoint model state must contain named tensors')
        if not torch.isfinite(value).all():
            raise ValueError('Non-finite checkpoint tensor: '+name)
        value = value.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(str(tuple(value.shape)).encode('ascii'))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _critic_layout(metadata):
    """Validate value-head meaning separately from actor parameter sharing.

    Earlier six-head files predate explicit layout metadata. Their five mixed
    heads and final legacy head retain that interpretation, including historical
    HAPPO files; they are never reinterpreted as the new scalar team critic.
    """
    dimension = metadata['critic_output_dim']
    if isinstance(dimension, (bool, np.bool_)) or not isinstance(dimension, (int, np.integer)):
        raise ValueError('Checkpoint critic output dimension must be an integer')
    explicit = 'critic_reward_layout' in metadata or 'actor_value_channels' in metadata
    if not explicit:
        if dimension != len(IDS)+1:
            raise ValueError('Scalar team checkpoints require explicit critic reward layout and actor channels')
        return 'mixed_by_reservoir_then_legacy_team', tuple(range(len(IDS))), int(dimension)
    layout = metadata.get('critic_reward_layout')
    channels = metadata.get('actor_value_channels')
    if (not isinstance(channels, (list, tuple)) or len(channels) != len(IDS) or
            any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                for value in channels)):
        raise ValueError('Checkpoint actor value channels must identify five integer heads')
    if layout == 'shared_team':
        expected_dimension, expected_channels, algorithm = 1, (0,)*len(IDS), 'happo'
    elif layout == 'mixed_by_reservoir_then_legacy_team':
        expected_dimension, expected_channels, algorithm = len(IDS)+1, tuple(range(len(IDS))), 'mappo'
    else:
        raise ValueError('Unsupported checkpoint critic reward layout')
    if dimension != expected_dimension or tuple(channels) != expected_channels:
        raise ValueError('Checkpoint critic reward layout, dimensions and actor channels disagree')
    if metadata.get('algorithm_name', 'happo') != algorithm:
        raise ValueError('Checkpoint algorithm disagrees with its critic reward layout')
    return layout, expected_channels, expected_dimension


def save_checkpoint(runner, path):
    """Exclusively save the completed policy, without changing runner or RNG state."""
    path = Path(path)
    if path.exists():
        raise FileExistsError('Checkpoint already exists: '+str(path))
    if runner.update_count < 1:
        raise ValueError('A checkpoint requires a completed policy update')
    if len(runner.actor) != len(IDS):
        raise ValueError('Checkpoint requires five actors')
    layout_metadata = dict(algorithm_name=getattr(runner, 'algorithm_name', 'happo'),
        critic_output_dim=runner.critic_output_dim, critic_reward_layout=runner.critic_reward_layout,
        actor_value_channels=list(runner.actor_value_channels))
    _critic_layout(layout_metadata)
    shared_actor = runner.algo_args['algo'].get('share_param', False)
    if not isinstance(shared_actor, bool):
        raise ValueError('Actor parameter sharing must be a boolean')
    if shared_actor:
        if getattr(runner, 'algorithm_name', 'happo') != 'mappo':
            raise ValueError('Only MAPPO checkpoints support a shared actor')
        first = runner.actor[0]
        core = getattr(first, 'core', None)
        if (core is None or any(actor.actor is not first.actor for actor in runner.actor) or
                [getattr(actor, 'agent_id', None) for actor in runner.actor] != list(range(len(IDS))) or
                any(getattr(actor, 'core', None) is not core for actor in runner.actor)):
            raise ValueError('Shared MAPPO checkpoint requires five ordered views of one actor core')
        raw_dim = int(runner.envs[0].observation_space[0].shape[0])
        actor_dim = raw_dim+len(IDS)
        if tuple(core.obs_space.shape) != (actor_dim,):
            raise ValueError('Shared actor input must append five agent identity features')
        actor_states = [_state_copy(first.actor)]
    else:
        actor_states = [_state_copy(actor.actor) for actor in runner.actor]
    critic_state = _state_copy(runner.critic.critic)
    normalizer_state = (None if runner.value_normalizer is None else
                        _state_copy(runner.value_normalizer))
    data_hashes = dict(runner.data.hashes)
    if any(_file_hash(runner.data.root/name) != expected for name, expected in data_hashes.items()):
        raise ValueError('Source dataset changed after it was loaded')
    settings = _plain(dict(physical=runner.physical_config, algorithm=runner.algo_args))
    metadata = _plain(dict(seed=runner.seed, policy_update=runner.update_count,
        policy_episode=runner.completed_episodes, training_steps=runner.training_steps,
        episode_horizon_days=runner.length, action_bins=runner.action_bins,
        reservoir_ids=list(IDS), **layout_metadata,
        observation=runner.envs[0].observation_metadata(), settings=settings,
        settings_sha256=_digest(settings), data_root=runner.data.root,
        data_hashes=data_hashes, source_hashes=_source_hashes(),
        training_years=list(runner.data.training_years),
        evaluation_years=list(runner.data.evaluation_years),
        evaluation_labels=list(runner.data.evaluation_labels),
        model_hashes=dict(actors=[_state_hash(s) for s in actor_states],
            critic=_state_hash(critic_state),
            value_normalizer=None if normalizer_state is None else _state_hash(normalizer_state)),
        purpose='deterministic evaluation replay; training resume is not implemented',
        torch_version=str(torch.__version__), numpy_version=str(np.__version__)))
    if shared_actor:
        metadata.update(actor_parameter_sharing=True, agent_ids=list(range(len(IDS))),
                        actor_observation_dim=actor_dim, raw_actor_observation_dim=raw_dim)
    if getattr(runner, 'credit_assignment', 'team_gae') == 'counterfactual':
        metadata.update(happo_credit_assignment='counterfactual',
                        actor_advantage_estimator='one_step_counterfactual')
    payload = dict(format=CHECKPOINT_FORMAT, version=CHECKPOINT_VERSION,
        metadata=metadata, metadata_sha256=_digest(metadata), actors=actor_states,
        critic=critic_state, value_normalizer=normalizer_state)
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with path.open('xb') as stream:
            created = True
            torch.save(payload, stream)
    except FileExistsError:
        raise
    except BaseException:
        # Only remove this invocation's incomplete, exclusively-created output.
        if created:
            path.unlink()
        raise
    return dict(path=str(path.resolve()), sha256=_file_hash(path),
        format=CHECKPOINT_FORMAT, version=CHECKPOINT_VERSION,
        policy_update=int(runner.update_count), policy_episode=int(runner.completed_episodes),
        training_steps=int(runner.training_steps), episode_horizon_days=int(runner.length),
        action_bins=int(runner.action_bins), source_hashes=metadata['source_hashes'],
        data_hashes=data_hashes, settings_sha256=metadata['settings_sha256'])


class _FrozenActor:
    def __init__(self, args, observation_space, action_space, state):
        self.args = copy.deepcopy(args)
        self.actor = StochasticPolicy(args, observation_space, action_space)
        self.actor.load_state_dict(state, strict=True)
        self.actor.requires_grad_(False).eval()

    def prep_rollout(self):
        self.actor.eval()

    def act(self, obs, available_actions=None, deterministic=True):
        return self.actor(obs, available_actions, deterministic)[0]

    def evaluate_actions(self, obs, action, available_actions=None, active_masks=None,
                         return_entropy_per_sample=False):
        return self.actor.evaluate_actions(obs, action, available_actions, active_masks,
                                          return_entropy_per_sample=return_entropy_per_sample)


def load_checkpoint(path, data_root=None):
    """Validate source/data/settings and reconstruct frozen CPU inference models.

    No runner, environment worker, optimizer, sampler, or training resume state is
    constructed. A relocated data directory is accepted only for matching bytes.
    """
    kwargs = dict(map_location='cpu')
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = True
    payload = torch.load(Path(path), **kwargs)
    if (not isinstance(payload, dict) or payload.get('format') != CHECKPOINT_FORMAT or
            payload.get('version') != CHECKPOINT_VERSION):
        raise ValueError('Unsupported cascade inference checkpoint format/version')
    try:
        metadata = payload['metadata']
        if _digest(metadata) != payload['metadata_sha256']:
            raise ValueError('Checkpoint metadata hash mismatch')
        settings = metadata['settings']
        if _digest(settings) != metadata['settings_sha256']:
            raise ValueError('Checkpoint effective settings hash mismatch')
        if metadata['source_hashes'] != _source_hashes():
            raise ValueError('Checkpoint implementation source/config hash mismatch')
        if metadata['reservoir_ids'] != list(IDS):
            raise ValueError('Checkpoint reservoir IDs disagree with the five-reservoir cascade')
        _, _, critic_output_dim = _critic_layout(metadata)
        shared_actor = metadata.get('actor_parameter_sharing', False)
        if not isinstance(shared_actor, bool):
            raise ValueError('Checkpoint actor parameter sharing must be a boolean')
        args = settings['algorithm']
        credit_mode = args['algo'].get('happo_credit_assignment','team_gae')
        if credit_mode not in ('team_gae','counterfactual'):
            raise ValueError('Checkpoint has unsupported actor credit assignment')
        if (metadata.get('happo_credit_assignment','team_gae') != credit_mode or
                (credit_mode == 'counterfactual' and (metadata.get('algorithm_name') != 'happo' or
                 metadata.get('actor_advantage_estimator') != 'one_step_counterfactual'))):
            raise ValueError('Checkpoint actor credit metadata disagrees with its settings')
        configured_sharing = args['algo'].get('share_param', False)
        if not isinstance(configured_sharing, bool) or configured_sharing != shared_actor:
            raise ValueError('Checkpoint actor-sharing architecture disagrees with its settings')
        if shared_actor:
            if (metadata.get('algorithm_name') != 'mappo' or
                    metadata.get('agent_ids') != list(range(len(IDS)))):
                raise ValueError('Shared checkpoint requires MAPPO and the five ordered agent IDs')
            if len(payload['actors']) != 1:
                raise ValueError('Shared checkpoint requires one actor state')
        elif len(payload['actors']) != 5:
            raise ValueError('Checkpoint requires five actor states')
        hashes = metadata['model_hashes']
        if (hashes['actors'] != [_state_hash(s) for s in payload['actors']] or
                hashes['critic'] != _state_hash(payload['critic']) or
                hashes['value_normalizer'] != (None if payload['value_normalizer'] is None else
                                               _state_hash(payload['value_normalizer']))):
            raise ValueError('Checkpoint model state hash mismatch')
        days = positive_integer(metadata['episode_horizon_days'], 'saved episode horizon')
        if (args['train']['episode_length'] != days or
                args['model']['action_bins'] != metadata['action_bins'] or
                bool(args['train']['use_valuenorm']) != (payload['value_normalizer'] is not None)):
            raise ValueError('Checkpoint model/horizon settings are inconsistent')
        validate_discount_factor(args['algo']['gamma'])
        root = Path(metadata['data_root'] if data_root is None else data_root).resolve()
        for name, expected in metadata['data_hashes'].items():
            if Path(name).name != name or _file_hash(root/name) != expected:
                raise ValueError('Checkpoint dataset hash mismatch: '+name)
        data = CascadeData(root, settings['physical'], args['experiment'])
        env = episode_environment(data, settings['physical'],
            [data.training_years[0]], 1, metadata['action_bins'])
        if (data.hashes != metadata['data_hashes'] or
                list(data.training_years) != metadata['training_years'] or
                list(data.evaluation_years) != metadata['evaluation_years'] or
                list(data.evaluation_labels) != metadata['evaluation_labels']):
            raise ValueError('Checkpoint dataset/split provenance mismatch')
        if env.observation_metadata() != metadata['observation']:
            raise ValueError('Checkpoint observation layout mismatch')
        if shared_actor:
            raw_dim = int(env.observation_space[0].shape[0])
            actor_dim = raw_dim+len(IDS)
            if (metadata.get('raw_actor_observation_dim') != raw_dim or
                    metadata.get('actor_observation_dim') != actor_dim):
                raise ValueError('Shared checkpoint actor observation dimensions disagree with agent IDs')
        network_args = {**args['model'], **args['algo'], 'critic_output_dim': critic_output_dim}
        # Initializers consume RNG, so reconstruction must restore its caller's
        # state even on a malformed model state/load failure.
        with torch.random.fork_rng(devices=[]):
            if shared_actor:
                from gym.spaces import Box
                from harl.algorithms.actors.shared_mappo import SharedMAPPOAgent
                core = _FrozenActor(network_args,
                    Box(-np.inf, np.inf, (actor_dim,), dtype=np.float32),
                    env.action_space[0], payload['actors'][0])
                core.obs_space = Box(-np.inf, np.inf, (actor_dim,), dtype=np.float32)
                actors = [SharedMAPPOAgent(core, i, num_agents=len(IDS)) for i in range(len(IDS))]
            else:
                actors = [_FrozenActor(network_args, env.observation_space[i], env.action_space[i], state)
                          for i, state in enumerate(payload['actors'])]
            critic = VNet(network_args, env.share_observation_space[0])
            critic.load_state_dict(payload['critic'], strict=True)
            critic.requires_grad_(False).eval()
            normalizer = None
            if payload['value_normalizer'] is not None:
                normalizer = ValueNorm(critic_output_dim)
                normalizer.load_state_dict(payload['value_normalizer'], strict=True)
                normalizer.requires_grad_(False).eval()
    except (KeyError, TypeError, RuntimeError) as exc:
        raise ValueError('Malformed cascade inference checkpoint: '+str(exc)) from exc
    return FrozenCascadePolicy(metadata, data, actors, critic, normalizer)


class ReplayTraceWriter:
    """Write full-precision, five-row-per-day replay measurements to an open CSV."""
    fields = REPLAY_TRACE_FIELDS

    def __init__(self, stream):
        self.stream = stream
        self.writer = csv.DictWriter(stream, fieldnames=self.fields, lineterminator='\n')
        self.writer.writeheader()
        stream.flush()

    def write(self, record):
        if set(record) != set(self.fields):
            raise ValueError('Replay trace rows must contain exactly REPLAY_TRACE_FIELDS')
        for name, value in record.items():
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                raise ValueError('Non-finite replay trace field: '+name)
        self.writer.writerow(record)

    def flush(self):
        self.stream.flush()


class FrozenCascadePolicy:
    """A saved policy reacting to current state, with optional isolated commands."""
    def __init__(self, metadata, data, actors, critic, value_normalizer):
        self.metadata = copy.deepcopy(metadata)
        self.data, self.actors, self.critic = data, actors, critic
        self.actor = actors
        self.value_normalizer = value_normalizer
        self.physical_config = copy.deepcopy(metadata['settings']['physical'])
        self.episode_days = metadata['episode_horizon_days']
        self.action_bins = metadata['action_bins']
        self.gamma = metadata['settings']['algorithm']['algo']['gamma']
        self.gae_lambda = validate_discount_factor(metadata['settings']['algorithm']['algo']['gae_lambda'])
        self.critic_reward_layout, self.actor_value_channels, self.critic_output_dim = _critic_layout(metadata)

    def _values(self, state):
        values = self.critic(state[None]).cpu().numpy()
        if self.value_normalizer is not None:
            values = self.value_normalizer.denormalize(values)
        # Preserve the network's float32 predictions, then promote diagnostic
        # arithmetic so vector GAE and the serialized scalar TD rows agree.
        # NumPy otherwise multiplies float32 arrays by gamma in float32 while
        # the corresponding scalar expression is promoted to float64.
        return values[0].astype(np.float64, copy=True)

    @torch.no_grad()
    def replay(self, source_years, episode_days=None, writer=None, action_overrides=None):
        days = self.episode_days if episode_days is None else positive_integer(episode_days, 'episode_days')
        if days != self.episode_days:
            raise ValueError('Replay horizon must equal the saved episode horizon')
        years = list(source_years)
        if (not years or any(isinstance(y, (bool, np.bool_)) or not isinstance(y, (int, np.integer))
                             for y in years) or len(set(years)) != len(years)):
            raise ValueError('Replay source years must be nonempty unique integer years')
        years = [int(y) for y in years]
        if set(years).issubset(self.data.training_years):
            phase = 'training_evaluation'
        elif set(years).issubset(self.data.evaluation_years):
            phase = 'evaluation'
        else:
            raise ValueError('Replay cannot mix training and evaluation years or use unknown years')
        overrides = {} if action_overrides is None else dict(action_overrides)
        for key, index in overrides.items():
            if (not isinstance(key, tuple) or len(key) != 2 or
                    isinstance(key[0], (bool, np.bool_)) or not isinstance(key[0], (int, np.integer)) or
                    not 1 <= key[0] <= days or key[1] not in IDS or
                    isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or
                    not 0 <= index < self.action_bins):
                raise ValueError('Overrides require (one-based step, reservoir ID): valid discrete index')
        env = episode_environment(self.data, self.physical_config, years, days, self.action_bins)
        initial = env.storage.copy()
        limits, metrics = DailyLimitMetrics(env), EpisodeRewardMetrics(self.gamma)
        rows, rewards, value_history, next_value_history = [], [], [], []
        mixed_rewards, team_rewards, legacy_rewards = [], [], []
        energy, spill, residual = 0., 0., 0.
        energy_by, spill_by = np.zeros(5), np.zeros(5)
        for step in range(1, days+1):
            observations, values = env.actor_observations(), self._values(env.state())
            policy_actions, commands, distributions = [], [], []
            for i, actor in enumerate(self.actors):
                _, _, distribution = actor.evaluate_actions(
                    observations[i:i+1], np.zeros((1, 1), dtype=np.int64))
                action = int(distribution.mode().item())
                policy_actions.append(action)
                commands.append(int(overrides.get((step, IDS[i]), action)))
                distributions.append(distribution)
            try:
                env.step(np.asarray(commands))
            except InfeasibleCascade as exc:
                _attach_evaluation_progress(exc, env, metrics, energy)
                raise
            transition = env.last_transition
            next_values = np.zeros(self.critic_output_dim) if step == days else self._values(env.state())
            metrics.update(transition)
            limits.update()
            energy += float(transition['energy'].sum())
            spill += float(transition['spill'].sum()*env.physics.dt)
            energy_by += transition['energy']
            spill_by += transition['spill']*env.physics.dt
            residual = max(residual, float(np.abs(transition['residual']).max()))
            team = self.critic_reward_layout == 'shared_team'
            rewards.append(np.asarray([transition['system_reward']]) if team else
                           np.r_[transition['mixed_rewards'], transition['reward']])
            value_history.append(values)
            next_value_history.append(next_values)
            mixed_rewards.append(np.asarray(transition['mixed_rewards']).copy())
            team_rewards.append(float(transition['system_reward']))
            legacy_rewards.append(float(transition['reward']))
            for i, name in enumerate(IDS):
                channel = self.actor_value_channels[i]
                learning_reward = rewards[-1][channel]
                td = float(learning_reward+self.gamma*next_values[channel]-values[channel])
                distribution = distributions[i]
                shortfalls = transition.get('service_shortfalls')
                integrals = transition.get('service_shortfall_integrals')
                row = dict(phase=phase, seed=self.metadata['seed'], policy_update=self.metadata['policy_update'],
                    episode_step=step, date=transition['date'], reservoir=name,
                    policy_action_index=policy_actions[i], command_action_index=commands[i],
                    effective_action_index=int(transition['actions'][i]),
                    action_overridden=int((step, name) in overrides), controllable=int(transition['controls'][i]),
                    normalized_action=float(transition['normalized_actions'][i]),
                    mapping_fraction=float(transition['normalized_actions'][i]**2),
                    policy_action_probability=float(distribution.probs[0, policy_actions[i]]),
                    command_probability=float(distribution.probs[0, commands[i]]),
                    command_log_probability=float(distribution.log_probs(torch.tensor([[commands[i]]])).item()),
                    entropy=float(distribution.entropy().item()),
                    local_generation_reward=float(transition['local_generation_rewards'][i]),
                    local_reward_penalty=float(transition['local_reward_penalties'][i]),
                    local_reward=float(transition['local_rewards'][i]), global_reward=float(transition['global_reward']),
                    mixed_reward=float(transition['mixed_rewards'][i]), system_reward=float(transition['system_reward']),
                    legacy_team_reward=float(transition['reward']),
                    critic_reward_layout=self.critic_reward_layout, value_channel=channel,
                    value_basis='shared_team_system_reward' if team else 'own_mixed_reward',
                    learning_reward=float(learning_reward), value=float(values[channel]),
                    next_value=float(next_values[channel]), td_residual=td,
                    realized_learning_reward_to_go=0., replay_gae_target=0., replay_gae_advantage=0.,
                    shared_team_value=float(values[0]) if team else None,
                    next_shared_team_value=float(next_values[0]) if team else None,
                    shared_team_td_residual=td if team else None,
                    realized_shared_team_reward_to_go=0.,
                    legacy_team_value=None if team else float(values[-1]), realized_mixed_reward_to_go=0.,
                    realized_legacy_team_reward_to_go=0.,
                    ecological_flow_shortfall_m3s=None if shortfalls is None else float(shortfalls['ecological_flow'][i]),
                    ecological_flow_deficit_m3=None if integrals is None else float(integrals['ecological_flow'][i]))
                for field, source in (
                    ('storage_start_m3', 'storage_start'), ('storage_end_m3', 'storage_end'),
                    ('level_start_m', 'levels_start'), ('level_end_m', 'levels_end'),
                    ('external_inflow_m3s', 'external'), ('inflow_m3s', 'inflow'), ('release_m3s', 'release'),
                    ('turbine_flow_m3s', 'turbine'), ('spill_m3s', 'spill'), ('net_head_m', 'net_head_m'),
                    ('power_MW', 'power'), ('energy_MWh', 'energy'), ('release_lower_m3s', 'low'),
                    ('release_upper_m3s', 'high'), ('storage_lower_m3', 'storage_lower_m3'),
                    ('storage_upper_m3', 'storage_upper_m3'), ('next_state_level_lower_m', 'next_state_level_lower_m'),
                    ('next_state_level_upper_m', 'next_state_level_upper_m'), ('balance_residual_m3', 'residual')):
                    row[field] = float(transition[source][i])
                row['spill_volume_m3'] = float(transition['spill'][i]*env.physics.dt)
                rows.append(row)
        remaining = np.zeros(self.critic_output_dim)
        remaining_mixed, remaining_team, remaining_legacy = np.zeros(len(IDS)), 0., 0.
        replay_gae = np.zeros(self.critic_output_dim)
        for t in reversed(range(days)):
            remaining = rewards[t]+self.gamma*remaining
            remaining_mixed = mixed_rewards[t]+self.gamma*remaining_mixed
            remaining_team = team_rewards[t]+self.gamma*remaining_team
            remaining_legacy = legacy_rewards[t]+self.gamma*remaining_legacy
            delta = rewards[t]+self.gamma*next_value_history[t]-value_history[t]
            replay_gae = delta+self.gamma*self.gae_lambda*replay_gae
            for i in range(5):
                row, channel = rows[5*t+i], self.actor_value_channels[i]
                row['realized_learning_reward_to_go'] = float(remaining[channel])
                row['replay_gae_advantage'] = float(replay_gae[channel])
                row['replay_gae_target'] = float(replay_gae[channel]+value_history[t][channel])
                row['realized_shared_team_reward_to_go'] = float(remaining_team)
                row['realized_mixed_reward_to_go'] = float(remaining_mixed[i])
                row['realized_legacy_team_reward_to_go'] = float(remaining_legacy)
        feasible = limits.finish()
        if writer is not None:
            for row in rows:
                writer.write(row)
            if hasattr(writer, 'flush'):
                writer.flush()
        used_years = list(dict.fromkeys(d.year for d in env.data.dates))
        totals = metrics.summary(detailed=True)
        totals.update(reward_metadata(self.data))
        totals['mixed_reward_definition'] = totals['actor_reward_definition']
        team = self.critic_reward_layout == 'shared_team'
        if team:
            totals['actor_reward_definition'] = SYSTEM_REWARD_DEFINITION
        totals['advantage_source'] = 'shared_team' if team else 'individual_mixed'
        totals['advantage_normalization'] = ('once_over_common_rollout_rows' if team else
            'per_reservoir_over_common_rollout_rows' if self.metadata['settings']['algorithm']['algo'].get(
                'normalize_shared_advantages', False) else 'per_reservoir_over_controllable_rows')
        return dict(status='ok', phase=phase, seed=self.metadata['seed'],
            policy_update=self.metadata['policy_update'], policy_episode=self.metadata['policy_episode'],
            training_steps=self.metadata['training_steps'], source_years=used_years,
            episode_horizon_days=days, completed_steps=days,
            critic_reward_layout=self.critic_reward_layout, critic_output_dim=self.critic_output_dim,
            actor_value_channels=list(self.actor_value_channels),
            start_date=env.data.dates[0].isoformat(), end_date=env.data.dates[-1].isoformat(),
            energy_MWh=energy, energy_TWh=energy/1e6, spill_volume_m3=spill,
            energy_by_reservoir_TWh=(energy_by/1e6).tolist(),
            spill_by_reservoir_billion_m3=(spill_by/1e9).tolist(),
            initial_storage_m3=initial.tolist(), final_storage_m3=env.storage.tolist(),
            total_storage_depletion_billion_m3=float((initial-env.storage).sum()/1e9),
            max_balance_residual_m3=residual, **feasible, **env.observation_metadata(),
            **totals,
            reward_penalties=dict(env.reward_settings),
            action_overrides=[dict(episode_step=int(step), reservoir=name, action_index=int(index))
                              for (step, name), index in sorted(overrides.items())],
            trace_rows=len(rows), value_interpretation='saved frozen critic in reward units; '
                'value_basis and value_channel identify the actual training reward; '
                'mixed/system/legacy Monte Carlo accounting is independent of critic heads; '
                'replay GAE uses this evaluation trajectory and does not reconstruct training advantages')
