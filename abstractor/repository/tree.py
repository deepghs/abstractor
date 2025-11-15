import io
import json
import logging
import os
import pathlib
import re
from functools import partial
from pprint import pformat
from typing import Optional, Union, List

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
from ..openai import ask_llm


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


_TASK_TYPES: List[str] = [
    'Audio-Text-to-Text',
    'Image-Text-to-Text',
    'Visual Question Answering',
    'Document Question Answering',
    'Video-Text-to-Text',
    'Visual Document Retrieval',
    'Any-to-Any',

    'Depth Estimation',
    'Image Classification',
    'Object Detection',
    'Image Segmentation',
    'Text-to-Image',
    'Image-to-Text',
    'Image-to-Image',
    'Image-to-Video',
    'Unconditional Image Generation',
    'Video Classification',
    'Text-to-Video',
    'Zero-Shot Image Classification',
    'Mask Generation',
    'Zero-Shot Object Detection',
    'Text-to-3D',
    'Image-to-3D',
    'Image Feature Extraction',
    'Keypoint Detection',
    'Video-to-Video',

    'Text Classification',
    'Token Classification',
    'Table Question Answering',
    'Question Answering',
    'Zero-Shot Classification',
    'Translation',
    'Summarization',
    'Feature Extraction',
    'Text Generation',
    'Fill-Mask',
    'Sentence Similarity',
    'Text Ranking',

    'Text-to-Speech',
    'Text-to-Audio',
    'Automatic Speech Recognition',
    'Audio-to-Audio',
    'Audio Classification',
    'Voice Activity Detection',

    'Tabular Classification',
    'Tabular Regression',
    'Time Series Forecasting',

    'Reinforcement Learning',
    'Robotics',

    'Graph Machine Learning',
]

ML_FRAMEWORKS: List[str] = [
    "PyTorch",
    "TensorFlow",
    "JAX",
    "Safetensors",
    "Transformers",
    "PEFT",
    "TensorBoard",
    "GGUF",
    "Diffusers",
    "ONNX",
    "stable-baselines3",
    "sentence-transformers",
    "ml-agents",
    "MLX",
    "TF-Keras",
    "Keras",
    "Adapters",
    "setfit",
    "timm",
    "Transformers.js",
    "sample-factory",
    "Joblib",
    "OpenVINO",
    "Flair",
    "fastai",
    "BERTopic",
    "spaCy",
    "ESPnet",
    "NeMo",
    "Core ML",
    "LiteRT",
    "OpenCLIP",
    "Rust",
    "Scikit-learn",
    "fastText",
    "KerasHub",
    "Asteroid",
    "speechbrain",
    "AllenNLP",
    "llamafile",
    "Fairseq",
    "PaddlePaddle",
    "Stanza",
    "Habana",
    "PaddleOCR",
    "Graphcore",
    "pyannote.audio",
    "SpanMarker",
    "paddlenlp",
    "unity-sentis",
    "DDUF",
    "univa",

    'dghs-imgutils',
    'animetimm',
]

_SYSTEM_PROMPT = f"""
You are an expert AI assistant specialized in analyzing Hugging Face repositories. Your task is to analyze repository information (including README files, data samples, and metadata) and extract structured information in JSON format.

## Your Task
Analyze the provided Hugging Face repository information and return a JSON object with the following structure:

```json
{{
    "repo_id": "huggingface/repo",
    "repo_type": "model/dataset/space",
    "task_types": ["Task Type 1", "Task Type 2", ...],
    "libraries": ["Library 1", "Library 2", ...],
    "modality": ["Modality 1", "Modality 2", ...],
    "abstract": "Abstract of this repository, should contain approx 40-60 words",
    "bio": "One sentence to describe what this repository is for",
    "keywords": ["Keyword 1", "Keyword 2", "Keyword 3", ...],
    "is_ready_to_view": true/false,
    "is_clear_enough": true/false
}}
```

## Field Guidelines

**repo_type**: Determine if this is a "model", "dataset", or "space" repository.

**task_types**: 
- For model repositories: Identify the specific ML tasks this model can perform
- For dataset repositories: Identify ALL possible ML tasks this dataset can be used for
- For space repositories: Identify the tasks demonstrated or supported
- Available options include: {", ".join(_TASK_TYPES)}, and others as appropriate.

**libraries**: Identify the key libraries, frameworks, and tools used or supported. Common options include: transformers, PyTorch, TensorFlow, JAX, ONNX, scikit-learn, pandas, numpy, OpenCV, Pillow, datasets, accelerate, diffusers, tokenizers, safetensors, gradio, streamlit, fastapi, and others as appropriate.

**modality**: Identify all data modalities involved. Options include: Text, Image, Audio, Video, Tabular, Time Series, Graph, 3D, Multi-Modal, and others as appropriate.

**abstract**: Write a concise 40-60 word summary describing what this repository does, its functionality, and its value/purpose.

**bio**: Write ONE clear sentence that describes what this repository is used for.

**keywords**: Provide 3-5 relevant keywords that complement the abstract and help categorize the repository.

**is_ready_to_view**: Set to `true` if the repository has a comprehensive README that clearly guides users on how to use it. Set to `false` if the README is missing, incomplete, or unclear.

**is_clear_enough**: Set to `true` if the repository's purpose and functionality can be clearly determined from its name, content samples, and description, and if this aligns with any existing README. Set to `false` if the repository's purpose is ambiguous or unclear.

## Instructions
1. Analyze all provided information carefully
2. Be precise and specific in your classifications
3. For datasets, consider multiple potential use cases
4. Ensure your abstract is exactly 40-60 words
5. Make your bio concise but informative
6. Choose keywords that accurately represent the repository's domain and purpose
7. Be honest in your assessment of readiness and clarity
8. Return ONLY the JSON object, no additional text or explanation
"""


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

                    if fn.lower().endswith('.md') or fn.lower().endswith('.py'):
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


def ask_llm_for_hf_repo_info(repo_id: str, repo_type: RepoTypeTyping = 'dataset', revision: str = 'main',
                             hf_token: Optional[str] = None):
    prompt = get_repository_prompt(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        hf_token=hf_token
    )

    cnt = 0
    while cnt < 5:
        try:
            text = ask_llm(
                prompts=[
                    {
                        "role": "system",
                        "content": _SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            )
            return json.loads(text)
        except:
            cnt += 1
            if cnt > 5:
                raise
            logging.exception(f'Error on parsing ({cnt}/{5}) ...')
