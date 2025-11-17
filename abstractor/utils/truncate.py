import logging
from typing import List

import tiktoken


def truncated_by_token(text: str, max_tokens: int = 130000):
    encoder = tiktoken.get_encoding("cl100k_base")
    original_length = len(encoder.encode(text))
    if original_length <= max_tokens:
        return text

    l, r = 0, len(text)
    while l < r:
        m = (l + r + 1) // 2
        if len(encoder.encode(text[:m])) <= max_tokens:
            l = m
        else:
            r = m - 1

    logging.info(f'Text truncated {original_length!r} --> {len(encoder.encode(text[:l]))}')
    return text[:l]


def truncated_prompts_by_token(prompts: List[dict], max_tokens: int = 130000):
    encoder = tiktoken.get_encoding("cl100k_base")
    retval = []
    for i, prompt in enumerate(prompts):
        length = len(encoder.encode(prompt['content']))
        if length <= max_tokens:
            logging.info(f'Prompt #{i}, role: {prompt["role"]!r} kept.')
            retval.append(prompt)
        else:
            length = max_tokens
            logging.info(f'Prompt #{i}, role: {prompt["role"]!r} truncated.')
            retval.append({
                **prompt,
                'content': truncated_by_token(prompt['content'], max_tokens=length)
            })

        max_tokens -= length
        if max_tokens <= 0:
            logging.warning(f'Rest of the prompts ({i}/{len(prompts)}) get truncated.')
            break

    return retval
