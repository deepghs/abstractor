import os
from functools import lru_cache

from openai import OpenAI


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
