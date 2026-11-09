"""Versioned format and answer-F1 components; no extra failure penalty."""


def auxiliary_reward(result, traj, config):
    result = dict(result)
    base = result['score']
    failure = (traj.finish_reason != 'answer' or traj.invalid_request
               or not result['reward/format_pass']
               or bool(result.get('metrics/is_no_final_answer',0)))
    enabled = config.reward_version=='format_f1_v2'
    valid_format = float(not failure)
    format_component = config.format_reward_weight*valid_format if enabled else 0.
    f1_component = config.f1_reward_weight*base*valid_format if enabled else base
    total = format_component+f1_component
    result.update({'score':total,'reward/total':total,
                   'reward/base_outcome':base,'reward/auxiliary':format_component,
                   'reward/format_component':format_component,'reward/f1_component':f1_component,
                   'reward/valid_format':valid_format,
                   'reward/completion_failure':float(failure),
                   'reward/auxiliary_enabled':float(enabled)})
    return result
