"""Metrics for actual QA trajectories, with explicit denominators and lengths."""
from collections import Counter
import math
import re
from statistics import mean, median

from inference.scoring import score
from mini.common import reward
from mini.diagnostics import protocol_metrics


def wilson(successes, n, z=1.959963984540054):
    if n == 0:
        return None
    p=successes/n
    denominator=1+z*z/n
    center=(p+z*z/(2*n))/denominator
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return [max(0.,center-radius),min(1.,center+radius)]


def describe(values):
    values=sorted(v for v in values if v is not None)
    return {'observed_n':len(values), 'mean':mean(values) if values else None,
            'median':median(values) if values else None,
            'p95_nearest_rank':values[math.ceil(.95*len(values))-1] if values else None}


def trajectory(record):
    turns=record.get('turns',[])
    messages=[t for t in turns if t['role'] in ('assistant','tool')]
    text='\n'.join(t.get('content','') if t['role']=='assistant' else '<tool_response>'+t.get('content','')+'</tool_response>' for t in messages)
    corrected=text if text.lstrip().startswith(('<think>','<reason>')) else '<think>\n'+text
    format_pass,format_reason=reward._is_valid_sequence(corrected)
    qa=score(record.get('prediction') or '',record['gold_answers'])
    protocol=protocol_metrics(messages)
    attempts=protocol['tool_call_attempts']; invalid=protocol['invalid_tool_calls']
    pending_invalid=False
    nonzero=timeout=0
    for t in messages:
        if t['role']=='assistant':
            pending_invalid=protocol_metrics([t])['invalid_tool_calls']>0
        elif not t.get('synthetic'):
            result=t.get('tool_result') or {}
            code=result.get('exit_code')
            nonzero+=int(code is not None and code!=0)
            timeout+=int(bool(result.get('timed_out')))
            if code==-2 and not pending_invalid:
                invalid+=1
    error=(record.get('error') or '').lower()
    context_error=bool(re.search(r'context length|maximum context|exceeds?.*context|no room to generate',error))
    truncated=any(t.get('finish_reason')=='length' for t in turns) or context_error or bool(record.get('response_budget_exhausted'))
    no_answer=not bool((record.get('prediction') or '').strip())
    flags={'format_failure':not format_pass,'invalid_tool_request_or_command':invalid>0,
           'tool_nonzero_exit':nonzero>0,'tool_timeout':timeout>0,
           'no_final_answer':no_answer,'context_truncation':truncated,
           'turn_budget_exhausted':record.get('finish_reason')=='max_turns',
           'api_error':record.get('finish_reason')=='api_error',
           'terminal_parse_error':record.get('finish_reason')=='parse_error'}
    flags['response_budget_exhausted']=bool(record.get('response_budget_exhausted'))
    flags['observation_budget_overflow']=bool(record.get('observation_budget_overflow'))
    success=bool(qa['em']) and format_pass and not no_answer
    return {'em':bool(qa['em']),'f1':qa['f1'],'format_pass':format_pass,'format_reason':format_reason,
            'tool_calls':record.get('n_tool_calls',protocol['tool_calls']), 'tool_attempts':attempts,'invalid_tool_calls':invalid,
            'success':success,'abnormal':any(flags.values()),'anomaly_flags':flags,
            'generated_completion_tokens':record.get('generated_completion_tokens'),
            'context_growth_tokens':record.get('context_growth_tokens'),
            'assistant_content_tokens':record.get('assistant_content_tokens')}


def aggregate(records):
    if not records:
        raise ValueError('empty evaluation')
    rows=[trajectory(r) for r in records]
    n=len(rows);attempts=sum(r['tool_attempts'] for r in rows);invalid=sum(r['invalid_tool_calls'] for r in rows)
    successes=[r for r in rows if r['success']]
    flags=Counter(k for r in rows for k,v in r['anomaly_flags'].items() if v)
    result={'n':n,'exact_match':mean(r['em'] for r in rows),'token_f1':mean(r['f1'] for r in rows),
            'format_pass_rate':mean(r['format_pass'] for r in rows),'mean_tool_calls':mean(r['tool_calls'] for r in rows),
            'tool_calls':sum(r['tool_calls'] for r in rows),'tool_call_attempts':attempts,'invalid_tool_calls':invalid,
            'invalid_tool_call_rate':invalid/attempts if attempts else None,
            'abnormal_trajectory_rate':mean(r['abnormal'] for r in rows),
            'anomaly_rates_nonexclusive':{k:flags[k]/n for k in rows[0]['anomaly_flags']},
            'no_final_answer_rate':flags['no_final_answer']/n,'context_truncation_rate':flags['context_truncation']/n,
            'api_error_rate':flags['api_error']/n,
            'success_count':len(successes),'success_definition':'EM=1 AND official format gate passes AND nonempty final prediction; recovered tool errors do not disqualify',
            'success_generated_tokens':describe([r['generated_completion_tokens'] for r in successes]),
            'success_context_growth_tokens':describe([r['context_growth_tokens'] for r in successes]),
            'success_assistant_content_tokens':describe([r['assistant_content_tokens'] for r in successes]),
            'em_wilson_95ci':wilson(sum(r['em'] for r in rows),n),
            'format_pass_wilson_95ci':wilson(sum(r['format_pass'] for r in rows),n),
            'abnormal_wilson_95ci':wilson(sum(r['abnormal'] for r in rows),n),
            'metric_contract':{'em_f1':'official scoring, max over gold aliases; include every scheduled question including errors',
                'format_pass':'official _is_valid_sequence over recorded assistant/tool transcript, with the official missing-leading-think fix',
                'tool_calls':'actual executor invocations; invalid requests failing before execution are not counted',
                'anomaly_rate':'union of listed flags per trajectory; rates overlap and must not be added',
                'nonzero_exit':'includes exit=1, which may be a legitimate no-match; this flag is not synonymous with an incorrect answer',
                'lengths':'API completion-token count measures generated tokens; context growth includes chat headers and tool observations; content-only BPE is a separate fallback, never substituted for API tokens',
                'conditional_length':'pooled over successful trajectories, with observed_n; null if no successes/measurements'}}
    return result


def report(records):
    sources=sorted({r.get('source') or 'unknown' for r in records})
    by_source={s:aggregate([r for r in records if (r.get('source') or 'unknown')==s]) for s in sources}
    scalar=('exact_match','token_f1','format_pass_rate','mean_tool_calls','abnormal_trajectory_rate')
    return {'overall':aggregate(records),'by_source':by_source,
            'macro_equal_dataset_weight':{k:mean(r[k] for r in by_source.values()) for k in scalar},
            'overall_definition':'pooled question metrics; with 30/source these equal macro for listed scalars, but successful lengths and per-call rates use their own denominators'}


def attach_token_lengths(record,tokenizer,usage):
    messages=record.get('messages') or [{'role':t['role'],'content':t['content']} for t in record.get('turns',[]) if t['role'] in ('system','user','assistant','tool')]
    record['generated_completion_tokens']=usage['completion_tokens'] if usage['complete'] else None
    record['api_request_count']=usage['requests']
    record['assistant_content_tokens']=sum(len(tokenizer.encode(t.get('content',''),add_special_tokens=False)) for t in record.get('turns',[]) if t['role']=='assistant')
    try:
        ids=tokenizer.apply_chat_template(messages,tools=[],enable_thinking=True,tokenize=True,return_dict=False,add_generation_prompt=False)
        prefix=tokenizer.apply_chat_template(messages[:2],tools=[],enable_thinking=True,tokenize=True,return_dict=False,add_generation_prompt=False)
        record['context_growth_tokens']=len(ids)-len(prefix)
        record['final_context_tokens']=len(ids)
    except (ValueError,TypeError) as exc:
        record['context_growth_tokens']=None
        record['length_measurement_error']=str(exc)


class BudgetedCompletion:
    """Enforce a cumulative response budget before each official inference call.

    Tool execution/formatting stays upstream. An observation may consume the
    remaining budget; in that case no further model call is made.
    """
    def __init__(self,create,tokenizer,prompt_length=1024,response_length=8192,context_length=9216,seed=42):
        self.original=create;self.tokenizer=tokenizer
        self.prompt_length=prompt_length;self.response_length=response_length;self.context_length=context_length;self.seed=seed
        self.reset('')

    def reset(self,qid):
        self.initial_prompt=None;self.requests=0;self.completion_tokens=0;self.complete=True
        self.budget_exhausted=False;self.qid=qid

    def __call__(self,*args,**kwargs):
        from mini.common import digest
        prompt=len(self.tokenizer.apply_chat_template(kwargs['messages'],tools=[],enable_thinking=True,add_generation_prompt=True,tokenize=True,return_dict=False))
        if self.initial_prompt is None:
            self.initial_prompt=prompt
            if prompt>self.prompt_length:
                raise ValueError(f'initial prompt budget exceeded: {prompt}>{self.prompt_length}')
        remaining=self.response_length-(prompt-self.initial_prompt)
        allowed=min(remaining,self.context_length-prompt)
        if allowed<=0:
            self.budget_exhausted=True
            raise RuntimeError('mini response budget exhausted; no further model request issued')
        requested=kwargs.get('max_tokens')
        kwargs['max_tokens']=min(requested,allowed) if requested is not None else allowed
        kwargs['seed']=int(digest([self.seed,self.qid])[:8],16)
        kwargs['extra_body']={'chat_template_kwargs':{'enable_thinking':True}}
        self.requests+=1
        try:
            result=self.original(*args,**kwargs)
        except Exception:
            self.complete=False
            raise
        usage=getattr(result,'usage',None)
        completion=getattr(usage,'completion_tokens',None)
        if completion is None:
            self.complete=False
        else:
            self.completion_tokens+=int(completion)
        return result

    def usage(self):
        return {'complete':self.complete and self.requests>0,'completion_tokens':self.completion_tokens,'requests':self.requests}


def markdown(results):
    lines=['# Fixed benchmark evaluation','',
        'Success = EM=1 + official Format Pass + nonempty final answer. Errors remain in EM/F1 denominators.',
        'Generated length uses observed API completion tokens; growth length includes observations/chat headers. Missing measurements are shown as NA.','']
    def number(x):
        return 'NA' if x is None else f'{x:.4f}'
    for checkpoint,r in results.items():
        lines += [f'## {checkpoint}','','| Dataset | n | EM | F1 | Format Pass | Avg Tool Calls | Success gen tokens | Success growth tokens | Success n | Abnormal |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for source,m in [*r['by_source'].items(),('overall pooled',r['overall'])]:
            lines.append('| '+' | '.join([source,str(m['n']),number(m['exact_match']),number(m['token_f1']),number(m['format_pass_rate']),number(m['mean_tool_calls']),
                number(m['success_generated_tokens']['mean']),number(m['success_context_growth_tokens']['mean']),str(m['success_count']),number(m['abnormal_trajectory_rate'])])+' |')
        lines += ['', 'Macro equal dataset weight: `'+str(r['macro_equal_dataset_weight'])+'`','']
    return '\n'.join(lines)
