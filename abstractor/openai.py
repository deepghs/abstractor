import logging
import os
from functools import lru_cache
from typing import List, Optional, Union

from openai import OpenAI

from .utils import truncated_prompts_by_token


@lru_cache()
def get_client() -> OpenAI:
    """
    Create and return an OpenAI client instance.

    This function initializes an OpenAI client using environment variables
    for API key and base URL configuration. The result is cached to avoid
    creating multiple client instances.

    :return: Configured OpenAI client instance.
    :rtype: OpenAI
    :raises KeyError: If required environment variables are not set.

    Example::
        >>> client = get_client()
        >>> # Use client for API calls
    """
    return OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        base_url=os.environ['OPENAI_SITE'],
    )


def _prompt_wrap(prompts: Union[str, List[dict]]) -> List[dict]:
    if isinstance(prompts, str):
        return [{
            "role": "user",
            "content": prompts,
        }]
    else:
        return prompts


def ask_llm(prompts: Union[str, List[dict]], model_name: Optional[str] = None,
            max_tokens: int = 110000):
    model_name = model_name or os.environ.get('OPENAI_MODEL_NAME') or 'deepseek-reasoner'
    logging.info(f'Asking model {model_name!r} ...')
    response = get_client().chat.completions.create(
        model=model_name,
        messages=truncated_prompts_by_token(_prompt_wrap(prompts), max_tokens=max_tokens),
    )

    response_text = response.choices[0].message.content
    logging.info(f'Response from model {model_name!r}:\n{response_text}')
    return response_text
