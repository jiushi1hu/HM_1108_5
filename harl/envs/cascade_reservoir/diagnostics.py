"""Read-only training diagnostics; no optimizer steps or result files."""
import json
import numpy as np
import torch


def require_finite(name, value):
    if not np.isfinite(value).all():
        raise FloatingPointError('Non-finite diagnostic input: ' + name)


@torch.no_grad()
def parameter_snapshot(module):
    return {name: p.detach().clone() for name, p in module.named_parameters() if p.requires_grad}


@torch.no_grad()
def parameter_update(module, before):
    """Measure tensor updates and final backward gradients without modifying them."""
    unchanged, missing, zero = [], [], []
    delta_sq, grad_sq = 0., 0.
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        require_finite('parameter ' + name, p.detach().cpu().numpy())
        delta = p.detach() - before[name]
        delta_sq += float(delta.double().square().sum())
        if torch.equal(p, before[name]):
            unchanged.append(name)
        if p.grad is None:
            missing.append(name)
        else:
            require_finite('gradient ' + name, p.grad.detach().cpu().numpy())
            grad_sq += float(p.grad.double().square().sum())
            if torch.count_nonzero(p.grad).item() == 0:
                zero.append(name)
    return dict(parameter_delta_l2=delta_sq**.5, final_grad_norm_post_clip=grad_sq**.5,
                unchanged_parameter_tensors=unchanged, missing_gradient_tensors=missing,
                zero_gradient_tensors=zero)


def fit_metrics(target, prediction):
    target, prediction = np.asarray(target).ravel(), np.asarray(prediction).ravel()
    variance = float(np.var(target))
    return dict(rmse=float(np.sqrt(np.mean((prediction-target)**2))),
                bias=float(np.mean(prediction-target)),
                explained_variance=None if variance < 1e-10 else
                    float(1.-np.var(target-prediction)/variance))


def monte_carlo_returns(rewards, masks, gamma):
    """Realized discounted reward-to-go, with no critic bootstrap.

    The calendar rollout ends at a true terminal state. Masks also keep an
    internal terminal boundary from leaking rewards into the preceding episode.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    masks = np.asarray(masks, dtype=np.float64)
    if gamma == 1. and np.all(masks[1:-1] == 1.):
        return np.cumsum(rewards[::-1], axis=0)[::-1].copy()
    returns = np.empty_like(rewards)
    remaining = np.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        remaining = rewards[t] + gamma * masks[t+1] * remaining
        returns[t] = remaining
    return returns


@torch.no_grad()
def actor_distribution(actor, buffer):
    """Use the same stored action support as collection and optimization."""
    available = buffer.available_actions
    _, _, dist = actor.evaluate_actions(
        buffer.obs[:-1].reshape(-1, buffer.obs.shape[-1]),
        buffer.actions.reshape(-1, 1),
        None if available is None else available[:-1].reshape(-1, available.shape[-1]))
    return type(dist)(logits=dist.logits.clone())


@torch.no_grad()
def before_update(runner):
    cb = runner.critic_buffer
    for name in ('share_obs', 'value_preds', 'rewards', 'returns', 'masks', 'bad_masks'):
        require_finite('critic buffer ' + name, getattr(cb, name))
    for i, buffer in enumerate(runner.actor_buffer):
        for name in ('obs', 'actions', 'action_log_probs', 'active_masks', 'masks'):
            require_finite('actor {} buffer {}'.format(i, name), getattr(buffer, name))
    prediction = cb.value_preds[:-1].copy()
    next_prediction = cb.value_preds[1:].copy()
    if runner.value_normalizer is not None:
        prediction = runner.value_normalizer.denormalize(prediction)
        next_prediction = runner.value_normalizer.denormalize(next_prediction)
    mc = monte_carlo_returns(cb.rewards, cb.masks, cb.gamma)
    result = dict(mc=mc, prediction=prediction, next_prediction=next_prediction,
                gae_targets=cb.returns[:-1].copy(),
                actor_parameters=[parameter_snapshot(a.actor) for a in runner.actor],
                critic_parameters=parameter_snapshot(runner.critic.critic),
                distributions=[actor_distribution(a,b) for a,b in zip(runner.actor, runner.actor_buffer)])
    credit = getattr(runner, 'counterfactual_credit', None)
    if credit is not None:
        result['counterfactual_credit'] = {key:credit[key].copy() for key in
            ('raw_advantages','normalized_advantages','q_actual','baseline')}
    return result


@torch.no_grad()
def after_update(runner, before):
    cb = runner.critic_buffer
    from harl.envs.cascade_reservoir.learning_signal import critic_reward_spec, learning_signal_report
    spec = critic_reward_spec(runner, cb.rewards.shape[-1])
    fitted = slice(spec['fit_channel'], spec['fit_channel']+1)
    mc_start = float(before['mc'][0,:,spec['fit_channel']].mean())
    prediction_start = float(before['prediction'][0,:,spec['fit_channel']].mean())
    if spec['shared_team'] and 'episode_reward' not in runner.last_reward_metrics:
        raise ValueError('Shared-team diagnostics require separate legacy episode accounting')
    accounting_reward = float(runner.last_reward_metrics.get('episode_reward', mc_start))
    prediction = runner.critic.get_values(
        cb.share_obs[:-1].reshape(-1, cb.share_obs.shape[-1]))
    prediction = prediction.cpu().numpy().reshape(cb.rewards.shape)
    if runner.value_normalizer is not None:
        prediction = runner.value_normalizer.denormalize(prediction)
    require_finite('post-update values', prediction)
    result = dict(episode=runner.completed_episodes, gamma=float(cb.gamma),
        update=runner.update_count, training_steps=runner.training_steps,
        batch_episodes=cb.rewards.shape[1],
        window_starts=list(runner.batch_window_starts),
        source_years=[list(years) for years in runner.batch_source_years],
        episode_horizon_days=cb.rewards.shape[0],
        critic_reward_layout=spec['layout'], critic_fitted_reward_basis=spec['fitted_reward_basis'],
        actor_value_channels=list(spec['actor_channels']),
        legacy_team_value_available=spec['legacy_value_available'],
        predicted_start_return_reward_units=prediction_start,
        realized_sampled_return_reward_units=mc_start,
        episode_reward_basis='legacy_team_reward',
        episode_penalty=runner.last_reward_metrics['episode_penalty'],
        episode_generation_reward=runner.last_reward_metrics['episode_generation_reward'],
        episode_penalty_components={key:runner.last_reward_metrics['episode_penalty_'+key]
            for key in ('ecological_flow',)},
        episode_reward=accounting_reward,
        discounted_episode_reward=accounting_reward,
        episode_energy_TWh=runner.last_rollout['episode_energy_TWh'],
        critic_update=parameter_update(runner.critic.critic, before['critic_parameters']),
        critic_mc_pre=fit_metrics(before['mc'][..., fitted], before['prediction'][..., fitted]),
        critic_mc_post=fit_metrics(before['mc'][..., fitted], prediction[..., fitted]),
        critic_gae_post=fit_metrics(cb.returns[:-1, :, fitted], prediction[..., fitted]),
        actors=[])
    result['learning_signal'] = learning_signal_report(runner, before, prediction)
    for i,(actor,buffer,old) in enumerate(zip(runner.actor, runner.actor_buffer,before['distributions'])):
        new = actor_distribution(actor,buffer)
        update = parameter_update(actor.actor, before['actor_parameters'][i])
        active = torch.from_numpy(buffer.active_masks[:-1].reshape(-1)>0)
        if not active.any():
            result['actors'].append(dict(agent=i, active_samples=0, **update)); continue
        action = torch.from_numpy(buffer.actions.reshape(-1,1))
        log_ratio = (new.log_probs(action)-old.log_probs(action))[active]
        ratio = log_ratio.exp()
        kl = torch.distributions.kl_divergence(old,new)[active]
        grid = torch.linspace(0., 1., new.probs.shape[-1])
        mean = (new.probs[active]*grid).sum(-1, keepdim=True)
        second = (new.probs[active]*grid.square()).sum(-1, keepdim=True)
        variance = (second-mean.square()).clamp_min(0.)
        deterministic = new.mode()[active].float()/(new.probs.shape[-1]-1)
        top_two = new.probs[active].topk(2, dim=-1).values
        require_finite('actor {} ratios'.format(i), ratio.cpu().numpy())
        require_finite('actor {} KL'.format(i), kl.cpu().numpy())
        sampled = action[active]/(new.probs.shape[-1]-1)
        result['actors'].append(dict(agent=i, active_samples=int(active.sum()),
            **update,
            entropy=float(new.entropy()[active].mean()),
            normalized_entropy=float(new.entropy()[active].mean()/np.log(new.probs.shape[-1])),
            mean_max_action_probability=float(top_two[:,0].mean()),
            mean_top_two_probability_gap=float((top_two[:,0]-top_two[:,1]).mean()),
            mode_change_fraction=float((old.mode()[active] != new.mode()[active]).float().mean()),
            sampled_action_mean=float(sampled.mean()),
            sampled_action_std=float(sampled.std(unbiased=False)),
            sampled_boundary_fraction=float(((sampled<=.01)|(sampled>=.99)).float().mean()),
            policy_kl=float(kl.mean().clamp(min=0)),
            ratio_min=float(ratio.min()),ratio_max=float(ratio.max()),
            clip_fraction=float(((ratio<1.-actor.clip_param)|(ratio>1.+actor.clip_param)).float().mean()),
            mean_u=float(mean.mean()),std_u=float(variance.sqrt().mean()),
            modal_mapping_fraction=float((deterministic**2).mean()),
            expected_sampled_mapping_fraction=float((mean**2+variance).mean())))
    return result


def print_update_window(runner, window=10):
    rows=runner.diagnostics_history[-window:]
    def avg(key,metric):
        values=[r[key][metric] for r in rows if r[key][metric] is not None]
        return float(np.mean(values)) if values else None
    active=[a for r in rows for a in r['actors'] if a['active_samples']]
    summary=dict(episode=runner.completed_episodes,update=runner.update_count,training_steps=runner.training_steps,
        window_updates=len(rows),window_episodes=sum(r.get('batch_episodes',1) for r in rows),gamma=rows[-1]['gamma'],
        critic_reward_layout=rows[-1].get('critic_reward_layout', 'legacy_team'),
        critic_fitted_reward_basis=rows[-1].get('critic_fitted_reward_basis', 'legacy_team_reward'),
        legacy_team_value_available=rows[-1].get('legacy_team_value_available', True),
        window_starts=[year for r in rows for year in r['window_starts']],
        episode_horizon_days=rows[-1]['episode_horizon_days'],
        mc_rmse_pre=avg('critic_mc_pre','rmse'),mc_rmse_post=avg('critic_mc_post','rmse'),
        mc_explained_variance_post=avg('critic_mc_post','explained_variance'),
        gae_explained_variance_post=avg('critic_gae_post','explained_variance'),
        max_actor_policy_kl=max((a['policy_kl'] for a in active),default=0.),
        mean_actor_clip_fraction=float(np.mean([a['clip_fraction'] for a in active])) if active else 0.,
        ratio_min=min((a['ratio_min'] for a in active),default=1.),
        ratio_max=max((a['ratio_max'] for a in active),default=1.),
        last_actors=rows[-1]['actors'])
    for prefix in ('predicted_start_return_', 'realized_sampled_return_'):
        key = prefix+'reward_units'
        summary[key] = float(np.mean([r[key] for r in rows]))
    summary['last_episode_penalty'] = rows[-1]['episode_penalty']
    summary['last_episode_generation_reward'] = rows[-1]['episode_generation_reward']
    summary['last_episode_penalty_components'] = rows[-1]['episode_penalty_components']
    # Existing grad norms are measured BEFORE clipping, not applied update sizes.
    training=runner.training_history[-window:]
    summary['gradient_clip_threshold']=float(runner.algo_args['algo']['max_grad_norm'])
    summary['critic_grad_norm_pre_clip_mean']=float(np.mean([r['critic_critic_grad_norm'] for r in training]))
    summary['last_episode_reward']=rows[-1]['episode_reward']
    summary['last_discounted_episode_reward']=rows[-1]['discounted_episode_reward']
    summary['last_episode_energy_TWh']=rows[-1]['episode_energy_TWh']
    summary['last_critic_update']=rows[-1]['critic_update']
    if 'learning_signal' in rows[-1]:
        summary['last_learning_signal'] = rows[-1]['learning_signal']
    summary['last_training_metrics']={k:v for k,v in training[-1].items()
                                     if k.startswith('actor') or k.startswith('critic_')}
    summary['happo_factor_min']=min(r['actor{}_factor_min'.format(i)] for r in training for i in range(5))
    summary['happo_factor_max']=max(r['actor{}_factor_max'.format(i)] for r in training for i in range(5))
    print('DIAG_UPDATE '+json.dumps(summary),flush=True)
