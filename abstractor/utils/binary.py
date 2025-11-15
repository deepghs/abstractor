from fsspec.implementations.http import HTTPFileSystem

_TEXT_CHARS = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7f})


def _is_binary_string(data: bytes) -> bool:
    """
    Check if the given byte data contains binary characters.

    :param data: The byte data to check.
    :type data: bytes

    :return: True if the data contains binary characters, False otherwise.
    :rtype: bool
    """
    return bool(data.translate(None, _TEXT_CHARS))


def _take_sample(url, size: int = 1024) -> bytes:
    fs = HTTPFileSystem()
    with fs.open(url, mode="rb") as f:
        return f.read(size)


def is_binary_url(url) -> bool:
    return _is_binary_string(_take_sample(url))


def is_text_url(url) -> bool:
    return not _is_binary_string(_take_sample(url))
