import io
import logging
import os
import pathlib
import re
from functools import partial
from pprint import pformat
from typing import Optional, Union

from hbutils.string import format_tree
from hfutils.entry.tree import TreeItem
from hfutils.operate.base import RepoTypeTyping, list_files_in_repository, get_hf_client
from hfutils.utils import hf_normpath, get_file_type, hf_fs_path, FileItemType
from huggingface_hub import hf_hub_download
from huggingface_hub.hf_api import RepoFile
from natsort import natsorted

from .csv import sample_from_csv
from .jsonl import sample_from_jsonl
from .parquet import sample_from_parquet
from .tar import sample_from_tar


def _get_sample_fn(filename):
    filename_lower = filename.lower()
    if filename_lower.endswith(".csv"):
        return partial(sample_from_csv, delimiter=',')
    elif filename_lower.endswith(".jsonl") or filename_lower.endswith(".ndjson"):
        return sample_from_jsonl
    elif filename_lower.endswith(".parquet"):
        return sample_from_parquet
    elif filename_lower.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2")):
        return sample_from_tar
    elif filename_lower.endswith(".tsv"):
        return partial(sample_from_csv, delimiter='\t')  # TSV is CSV with tab delimiter
    # JSON format deliberately excluded - requires full file download

    return None


def _get_tree(repo_id: str, repo_type: RepoTypeTyping, dir_in_repo: str,
              revision: Optional[str] = None, show_all: bool = False, hf_token: Optional[str] = None) -> TreeItem:
    """
    Retrieve the tree structure of files in a HuggingFace repository.

    :param repo_id: The ID of the repository.
    :type repo_id: str
    :param repo_type: The type of the repository.
    :type repo_type: RepoTypeTyping
    :param dir_in_repo: The directory in the repository to start from.
    :type dir_in_repo: str
    :param revision: The revision of the repository to use.
    :type revision: Optional[str]
    :param show_all: Whether to show hidden files.
    :type show_all: bool

    :return: The root TreeItem representing the directory structure.
    :rtype: TreeItem
    """
    root = {}
    for filepath in list_files_in_repository(
            repo_id=repo_id,
            repo_type=repo_type,
            subdir=dir_in_repo,
            revision=revision,
            pattern=['**/*'],
            hf_token=hf_token,
    ):
        filename = hf_normpath(os.path.relpath(filepath, dir_in_repo))
        segments = re.split(r'[\\/]+', filename)
        if any(segment.startswith('.') and segment != '.' for segment in segments) and not show_all:
            continue

        current_node = root
        for i, segment in enumerate(segments):
            if segment not in current_node:
                if i == (len(segments) - 1):
                    current_node[segment] = get_file_type(segment)
                else:
                    current_node[segment] = {}
            current_node = current_node[segment]

    root_name = hf_fs_path(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=dir_in_repo,
        revision=revision,
    )

    def _recursion(cur_node: Union[dict, FileItemType], parent_name: str, is_exist: bool = False):
        if isinstance(cur_node, dict):
            return TreeItem(
                name=parent_name,
                type_=FileItemType.FOLDER,
                children=[
                    _recursion(cur_node=value, parent_name=name, is_exist=is_exist)
                    for name, value in natsorted(cur_node.items())
                ],
                exist=is_exist,
            )
        else:
            return TreeItem(
                name=parent_name,
                type_=cur_node,
                children=[],
                exist=is_exist,
            )

    exist = True
    if not root:
        hf_client = get_hf_client(hf_token=hf_token)
        paths = hf_client.get_paths_info(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            paths=[dir_in_repo],
        )
        if len(paths) == 0:
            exist = False
        elif len(paths) == 1:
            pathobj = paths[0]
            if isinstance(pathobj, RepoFile):  # the subdir is a file
                root = get_file_type(dir_in_repo)
        else:
            assert len(paths) == 1, \
                f'Multiple path {dir_in_repo!r} found in repo {root_name!r}, ' \
                f'this must be caused by HuggingFace API.'  # pragma: no cover

    return _recursion(
        cur_node=root,
        parent_name=root_name,
        is_exist=exist,
    )


def get_repository_prompt(repo_id: str, repo_type: RepoTypeTyping = 'dataset', revision: str = 'main',
                          hf_token: Optional[str] = None, max_items: Optional[int] = 10):
    tree_root = _get_tree(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        dir_in_repo='.',
        hf_token=hf_token,
    )

    with io.StringIO() as sf:
        print(f'# Directory Tree', file=sf)
        print(f'', file=sf)
        print(f'This is the directory tree of this repository:', file=sf)
        print(format_tree(
            tree_root,
            format_node=TreeItem.get_name,
            get_children=TreeItem.get_children,
        ), file=sf)
        print(f'', file=sf)

        print(f'# Sample Files', file=sf)
        print(f'', file=sf)
        print(f'These are some samples from the data files', file=sf)
        print(f'', file=sf)

        def _recursive(tree: TreeItem, filename: str):
            cnt = 0
            for item in tree.children:
                fn = hf_normpath(os.path.join(filename, item.name))
                if item.type_ == FileItemType.FOLDER:
                    _recursive(tree=item, filename=fn)
                else:
                    if max_items is not None and cnt >= max_items:
                        continue

                    if fn.lower().endswith('.md'):
                        logging.info(f'Loading text file {fn!r} ...')
                        print(f'## {fn}', file=sf)
                        print(f'', file=sf)
                        print(pathlib.Path(hf_hub_download(
                            repo_id=repo_id,
                            repo_type=repo_type,
                            filename=fn, revision=revision,
                            token=hf_token,
                        )).read_text(), file=sf)
                        print(f'', file=sf)
                        cnt += 1
                    else:
                        _sfn = _get_sample_fn(fn)
                        if _sfn is None:
                            continue

                        print(f'## {fn}', file=sf)
                        print(f'', file=sf)
                        print(pformat(_sfn(
                            repo_id=repo_id,
                            repo_type=repo_type,
                            filename=fn,
                            revision=revision,
                            hf_token=hf_token,
                        )), file=sf)
                        print(f'', file=sf)
                        cnt += 1

        _recursive(tree_root, filename='.')

        return sf.getvalue()
