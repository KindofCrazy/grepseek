def render_prefix(tokenizer, messages, end):
    if end == 0:
        return []
    if end == 1:
        rendered = tokenizer.apply_chat_template(messages[:2], tools=[], tokenize=False, add_generation_prompt=False, enable_thinking=True)
        system = rendered.split('<|im_start|>user\n', 1)[0]
        return tokenizer.encode(system, add_special_tokens=False)
    return tokenizer.apply_chat_template(messages[:end], tools=[], tokenize=True, return_dict=False, add_generation_prompt=False, enable_thinking=True)


def turn_tokens(tokenizer, messages, index):
    before = render_prefix(tokenizer, messages, index)
    after = render_prefix(tokenizer, messages, index + 1)
    if after[:len(before)] != before:
        raise ValueError('chat template changed earlier tokens; unsafe turn loss mask')
    ids = after[len(before):]
    mask = [0] * len(ids)
    if messages[index]['role'] == 'assistant':
        generation = tokenizer.apply_chat_template(messages[:index], tools=[], tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=True)
        header = generation[len(before):]
        if generation[:len(before)] != before:
            raise ValueError('assistant generation prefix does not match SFT target')
        # Empty thinking merges two newlines into one BPE token; the prompt's
        # final single newline then is not a token-prefix of the target. Mask
        # the shared header tokens, and supervise the boundary whitespace token.
        common = 0
        for a, b in zip(ids, header):
            if a != b:
                break
            common += 1
        role_header = tokenizer.encode('<|im_start|>assistant\n', add_special_tokens=False)
        if ids[:len(role_header)] != role_header or common < len(role_header):
            raise ValueError('assistant role header mismatch')
        mask[common:] = [1] * (len(ids) - common)
    return ids, mask


