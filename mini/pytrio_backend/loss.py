"""Masked, globally token-mean PPO objective; environment tokens have zero gradient."""
import numpy as np
import pytrio as trio
import torch


def ppo_batch(trajectories, clip):
    datums, metadata = [], []
    for traj in trajectories:
        mask = np.asarray(traj.mask[1:], dtype=bool)
        targets = np.asarray(traj.tokens[1:], dtype=np.int64)
        targets[~mask] = 0
        datums.append(trio.Datum(model_input=trio.ModelInput.from_ints(traj.tokens[:-1]),
                                 loss_fn_inputs={'target_tokens': targets}))
        metadata.append((mask, np.asarray(traj.old_logprobs[1:]), traj.advantage))
    denominator = sum(int(mask.sum()) for mask, _, _ in metadata)
    if not denominator:
        raise ValueError('No assistant tokens to optimize')

    def objective(data, logprobs):
        if len(logprobs) != len(metadata) or len(data) != len(metadata):
            raise ValueError('Custom loss batch alignment error')
        losses, ratios, deltas, clipped = [], [], [], []
        for new, (mask, old, advantage) in zip(logprobs, metadata):
            new = new.reshape(-1)
            if new.numel() != len(mask):
                raise ValueError('Custom loss token alignment error')
            selected = torch.as_tensor(mask, device=new.device)
            delta = new[selected] - torch.as_tensor(old[mask], device=new.device, dtype=new.dtype)
            if not torch.isfinite(delta).all() or (delta.abs() > 50).any():
                raise ValueError('Invalid or extremely stale on-policy logprobs')
            ratio = delta.exp()
            unclipped = ratio * advantage
            bounded = ratio.clamp(1-clip, 1+clip) * advantage
            losses.append(-torch.minimum(unclipped, bounded).sum()/denominator)
            ratios.append(ratio.detach())
            deltas.append(delta.detach())
            clipped.append(((ratio < 1-clip) | (ratio > 1+clip)).detach())
        loss = torch.stack(losses).sum()
        ratio, delta = torch.cat(ratios), torch.cat(deltas)
        return loss, {'policy_loss':float(loss.detach()), 'ratio_mean':float(ratio.mean()),
                      'approx_kl':float(((ratio-1)-delta).mean()),
                      'clip_fraction':float(torch.cat(clipped).float().mean()),
                      'loss_tokens':denominator}
    return datums, objective
