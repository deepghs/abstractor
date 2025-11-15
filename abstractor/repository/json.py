import json
import logging
from typing import Any, Dict, List, Optional

from fsspec.implementations.http import HTTPFileSystem
from hbutils.scale import size_to_bytes_str
from hbutils.string import plural_word
from hfutils.operate.base import RepoTypeTyping
from hfutils.utils import hf_fs_path

from ..utils import hf_get_resource_url


class JSONSummarizer:
    """
    A class for summarizing and truncating JSON structures to make them more manageable.

    This class provides functionality to limit the size of JSON objects by controlling
    the maximum number of items in collections, the depth of nested structures, and
    the length of string values. It's useful for creating readable summaries of large
    or complex JSON data structures.
    """

    def __init__(self, max_items: Optional[int] = 10, max_keys: Optional[int] = None,
                 max_depth: Optional[int] = None, max_string_length: Optional[int] = 150):
        """
        Initialize the JSONSummarizer with configuration parameters.

        :param max_items: Maximum number of items to show in lists. If None, no limit is applied.
        :type max_items: Optional[int]
        :param max_keys: Maximum number of keys to show in dictionaries. If None, no limit is applied.
        :type max_keys: Optional[int]
        :param max_depth: Maximum depth of nested structures to traverse. If None, no limit is applied.
        :type max_depth: Optional[int]
        :param max_string_length: Maximum length of strings before truncation. If None, no limit is applied.
        :type max_string_length: Optional[int]
        """
        self.max_items = max_items
        self.max_keys = max_keys
        self.max_depth = max_depth
        self.max_string_length = max_string_length

    def summarize(self, obj: Any, current_depth: int = 0) -> Any:
        """
        Recursively summarize a JSON structure by applying size and depth limitations.

        This method traverses the input object and applies the configured limits to create
        a summarized version. It handles dictionaries, lists, strings, and other data types
        appropriately based on the instance configuration.

        :param obj: The object to summarize (can be any JSON-serializable type).
        :type obj: Any
        :param current_depth: The current depth in the recursive traversal.
        :type current_depth: int

        :return: A summarized version of the input object.
        :rtype: Any
        """
        if self.max_depth is not None and current_depth >= self.max_depth:
            return f"<Depth Limitation: {type(obj).__name__}>"

        if isinstance(obj, dict):
            return self._summarize_dict(obj, current_depth)
        elif isinstance(obj, list):
            return self._summarize_list(obj, current_depth)
        elif isinstance(obj, str):
            return self._summarize_string(obj)
        else:
            return obj

    def _summarize_dict(self, obj: Dict, current_depth: int) -> Dict:
        """
        Summarize a dictionary by limiting the number of keys shown.

        This method processes a dictionary and includes only up to max_keys entries.
        If the dictionary has more keys than the limit, it adds an indicator showing
        how many additional keys were omitted.

        :param obj: The dictionary to summarize.
        :type obj: Dict
        :param current_depth: The current depth in the recursive traversal.
        :type current_depth: int

        :return: A summarized dictionary with limited keys.
        :rtype: Dict
        """
        result = {}
        items_count = 0

        for key, value in obj.items():
            if self.max_keys is not None and items_count >= self.max_keys:
                result[f"... +{plural_word(len(obj) - items_count, 'more key')}"] = "..."
                break
            else:
                result[key] = self.summarize(value, current_depth + 1)
                items_count += 1

        return result

    def _summarize_list(self, obj: List, current_depth: int) -> List:
        """
        Summarize a list by limiting the number of items shown.

        This method processes a list and includes only up to max_items entries.
        If the list has more items than the limit, it adds an indicator showing
        how many additional items were omitted.

        :param obj: The list to summarize.
        :type obj: List
        :param current_depth: The current depth in the recursive traversal.
        :type current_depth: int

        :return: A summarized list with limited items.
        :rtype: List
        """
        if len(obj) == 0:
            return []

        result = []
        show_count = len(obj)
        if self.max_items is not None:
            show_count = min(self.max_items, len(obj))

        for i in range(show_count):
            result.append(self.summarize(obj[i], current_depth + 1))

        if self.max_items is not None and len(obj) > self.max_items:
            result.append(f"... +{plural_word(len(obj) - self.max_items, 'more item')}")

        return result

    def _summarize_string(self, obj: str) -> str:
        """
        Summarize a string by truncating it if it exceeds the maximum length.

        This method checks if a string exceeds the configured maximum length and
        truncates it if necessary, adding an indicator of the original length.

        :param obj: The string to summarize.
        :type obj: str

        :return: The original string or a truncated version with length indicator.
        :rtype: str
        """
        if self.max_string_length is not None and len(obj) > self.max_string_length:
            return obj[:self.max_string_length] + f"... ({plural_word(len(obj), 'character')}) ..."
        return obj


def sample_from_json(repo_id: str, filename: str, repo_type: RepoTypeTyping = 'dataset', revision: str = 'main',
                     hf_token: Optional[str] = None, max_items: Optional[int] = 10, max_keys: Optional[int] = None,
                     max_depth: Optional[int] = None, max_string_length: Optional[int] = 150,
                     truncated_size: int = 1500):
    fs_path = hf_fs_path(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=filename,
        revision=revision,
    )
    logging.info(f'Sampling parquet file {fs_path!r} ...')
    url, content_size = hf_get_resource_url(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=filename,
        revision=revision,
        hf_token=hf_token,
    )

    fs = HTTPFileSystem()
    if content_size <= 20 * 1024 ** 2:
        with fs.open(url, mode="rb") as f:
            meta_info = json.load(f)

        meta_simplified = JSONSummarizer(
            max_items=max_items,
            max_keys=max_keys,
            max_depth=max_depth,
            max_string_length=max_string_length,
        ).summarize(meta_info)
        return {
            "json_simplified": meta_simplified,
            "simplified_size": len(json.dumps(meta_simplified)),
            "file_size": content_size,
        }

    else:
        with fs.open(url, mode="r") as f:
            prefix = f.read(truncated_size // 2)
            f.seek(content_size - truncated_size // 2)
            suffix = f.read(truncated_size // 2)
        simplified = f'{prefix} ... ({size_to_bytes_str(content_size, system="si", sigfigs=3)} in total) ... {suffix}'
        return {
            'simplified_str': simplified,
            'simplified_size': len(simplified),
            "file_size": content_size,
        }
