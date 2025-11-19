import logging
from typing import List, Optional

import tiktoken
from hbutils.string import plural_word


def _encode(text, encoder: Optional = None):
    encoder = encoder or tiktoken.get_encoding("cl100k_base")
    return encoder.encode(text, disallowed_special=())


def truncated_by_token(text: str, max_tokens: int = 130000):
    encoder = tiktoken.get_encoding("cl100k_base")
    original_length = len(_encode(text, encoder=encoder))
    if original_length <= max_tokens:
        return text

    l, r = 0, len(text)
    while l < r:
        m = (l + r + 1) // 2
        if len(_encode(text[:m], encoder=encoder)) <= max_tokens:
            l = m
        else:
            r = m - 1

    logging.info(f'Text truncated {original_length!r} --> {len(_encode(text[:l], encoder=encoder))}')
    return text[:l]


def truncated_prompts_by_token(prompts: List[dict], max_tokens: int = 130000):
    encoder = tiktoken.get_encoding("cl100k_base")
    retval = []
    for i, prompt in enumerate(prompts):
        length = len(_encode(prompt['content'], encoder=encoder))
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
            rest_count = len(prompts) - i - 1
            if rest_count > 0:
                logging.warning(f'Rest of the {plural_word(rest_count, "prompt")} get truncated.')
            break

    return retval
