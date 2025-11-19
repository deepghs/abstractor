from hbutils.string import plural_word

from ..utils import truncated_by_token


def sample_text(text: str, max_tokens: int = 30000):
    original_length = len(text)
    if original_length > max_tokens * 15:
        text = text[:max_tokens * 15]

    text = truncated_by_token(text, max_tokens=max_tokens)
    if len(text) < original_length:
        text += f'... ({plural_word(original_length, "character")} in total) ...'

    return text
