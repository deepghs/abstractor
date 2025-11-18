import io
import logging
import os
import pathlib
import re
import textwrap
import time
from functools import partial
from typing import Optional, Union, List

from hbutils.string import format_tree, plural_word
from hfutils.entry.tree import TreeItem
from hfutils.operate.base import RepoTypeTyping, list_files_in_repository, get_hf_client
from hfutils.utils import hf_normpath, get_file_type, hf_fs_path, FileItemType
from huggingface_hub import hf_hub_download
from huggingface_hub.hf_api import RepoFile
from natsort import natsorted

from .csv import sample_from_csv
from .json import sample_from_json
from .jsonl import sample_from_jsonl
from .parquet import sample_from_parquet
from .tar import sample_from_tar
from ..openai import ask_llm
from ..utils import parse_json_from_llm_output


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
    elif filename_lower.endswith('.json'):
        return sample_from_json
    # JSON format deliberately excluded - requires full file download

    return None


def get_hf_repo_tree(repo_id: str, repo_type: RepoTypeTyping, dir_in_repo: str,
                     revision: Optional[str] = None, show_all: bool = False,
                     hf_token: Optional[str] = None) -> TreeItem:
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
    "Ultralytics",

    "datasets",
    "accelerate",
    "diffusers",
    "tokenizers",
    "safetensors",

    "streamlit",
    "gradio",
    "fastapi",

    'dghs-imgutils',
    'animetimm',
    'cheesechaser',
]


def _tree_simple(tree: TreeItem, max_cnt: Optional[int] = 20):
    children = []
    child_cnt, child_cnt_added = 0, 0
    sub_cnt, sub_cnt_added = 0, 0
    for item in tree.children:
        if item.type_ == FileItemType.FOLDER:
            if max_cnt is not None and sub_cnt < max_cnt:
                children.append(_tree_simple(item))
                sub_cnt_added += 1
            sub_cnt += 1
        else:
            if max_cnt is not None and child_cnt < max_cnt:
                children.append(item)
                child_cnt_added += 1
            child_cnt += 1

    if sub_cnt > sub_cnt_added:
        children.append(TreeItem(
            name=f'... {plural_word(sub_cnt, "folder")} in total ...',
            type_=FileItemType.FOLDER,
            children=[],
            exist=True,
        ))
    if child_cnt > child_cnt_added:
        children.append(TreeItem(
            name=f'... {plural_word(child_cnt, "file")} in total ...',
            type_=FileItemType.FILE,
            children=[],
            exist=True,
        ))

    return TreeItem(
        name=tree.name,
        type_=tree.type_,
        children=children,
        exist=tree.exist,
    )


def ask_llm_for_hf_repo_key_files(repo_id: str, repo_type: RepoTypeTyping = 'dataset', revision: str = 'main',
                                  hf_token: Optional[str] = None, max_retries: int = 5, max_sample_count: int = 15):
    tree_root = get_hf_repo_tree(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        dir_in_repo='.',
        hf_token=hf_token,
    )

    _PICK_SYSTEM_PROMPT = textwrap.dedent(f"""
    You are a repository content sampling specialist. Your task is to analyze a given Hugging Face repository directory tree and intelligently select the most informative files for content analysis.

    **OBJECTIVE:**
    Given a repository path tree, select up to {max_sample_count} files that would provide the best understanding of what the repository contains and its purpose. Prioritize efficiency - select fewer files if they provide sufficient information.

    **SELECTION CRITERIA (in order of priority):**
    1. **Documentation files**: README.md, documentation, guides
    2. **Metadata/Configuration**: meta.json, config files, dataset cards
    3. **Sample data files**: Representative examples from different data categories
    4. **Code files**: Python scripts, processing code
    5. **Schema/Structure files**: Files that reveal data structure and format

    **CONSTRAINTS:**
    - Maximum {max_sample_count} files total
    - Only select files with these extensions: .md, .py, .parquet, .csv, .tsv, .json, .jsonl, .tar
    - Files must be downloadable via hf_hub_download (use exact repository paths)
    - Avoid redundant content - if files appear to contain similar information based on naming patterns, sample only representative examples
    - Prioritize smaller, more informative files over large data archives when possible
    - You must make sure your selected file is exist!!! To make sure about this, you can only select files that explicitly exist in the given directory tree, not the probably existing ones.

    **OUTPUT FORMAT:**
    Return a JSON array of strings, where each string is the exact file path as it appears in the repository tree.

    **EXAMPLE:**
    For a repository with README.md, config files, and numbered data batches, you might select:
    ```json
    ["README.md", "meta.json", "tables/page-1.parquet", "images/0/001.json"]
    ```

    **ANALYSIS APPROACH:**
    1. First identify documentation and metadata files
    2. Analyze the directory structure to understand data organization
    3. Select representative samples from each major data category
    4. Avoid selecting multiple files that appear to be sequential/similar batches
    5. Ensure selected files together provide comprehensive repository understanding
    6. You Must make sure your requested file is actually exist in this repository!!!! So please must use the existing file in my provided directory tree, not the truncated ones!!!!
    7. Return ONLY the JSON object, no additional text or explanation or prefix or something, just those simplest List[str]
    """).strip()

    with io.StringIO() as sf:
        print(f'Repo ID: {repo_id}', file=sf)
        print(f'Repo Type: {repo_type}', file=sf)
        print(f'', file=sf)
        print(f'# Directory Tree', file=sf)
        print(f'', file=sf)
        print(f'This is the directory tree of this repository:', file=sf)
        print(format_tree(
            _tree_simple(tree_root),
            format_node=TreeItem.get_name,
            get_children=TreeItem.get_children,
        ), file=sf)
        print(f'', file=sf)

        cnt = 0
        while True:
            try:
                expected_filenames = parse_json_from_llm_output(ask_llm([
                    {
                        "role": "system",
                        "content": _PICK_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": sf.getvalue(),
                    }
                ], model_name='deepseek-chat'))
            except:
                cnt += 1
                if cnt > max_retries:
                    raise
                logging.exception(f'Error on parsing ({cnt}/{max_retries}) ...')
            else:
                break

        logging.info(f'Expected filename: {expected_filenames!r}')
        return expected_filenames


def get_hf_repo_abstract_prompt(repo_id: str, repo_type: RepoTypeTyping = 'dataset', revision: str = 'main',
                                hf_token: Optional[str] = None, max_retries: int = 15, max_sample_count: int = 15):
    tree_root = get_hf_repo_tree(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        dir_in_repo='.',
        hf_token=hf_token,
    )

    with io.StringIO() as sf:
        print(f'Repo ID: {repo_id}', file=sf)
        print(f'Repo Type: {repo_type}', file=sf)
        print(f'', file=sf)
        print(f'# Directory Tree', file=sf)
        print(f'', file=sf)
        print(f'This is the directory tree of this repository:', file=sf)
        print(format_tree(
            _tree_simple(tree_root),
            format_node=TreeItem.get_name,
            get_children=TreeItem.get_children,
        ), file=sf)
        print(f'', file=sf)

        expected_filenames = ask_llm_for_hf_repo_key_files(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            hf_token=hf_token,
            max_retries=max_retries,
            max_sample_count=max_sample_count,
        )

        print(f'# Sample Files', file=sf)
        print(f'', file=sf)
        print(f'These are some samples from the data files', file=sf)
        print(f'', file=sf)

        for fn in expected_filenames:
            if fn.lower().endswith('.md') or fn.lower().endswith('.py'):
                logging.info(f'Loading text file {fn!r} ...')
                try:
                    _sfn_text = pathlib.Path(hf_hub_download(
                        repo_id=repo_id,
                        repo_type=repo_type,
                        filename=fn, revision=revision,
                        token=hf_token,
                    )).read_text()
                except:
                    logging.exception(f'Sample skipped for file {fn!r}...')
                    continue

                print(f'## {fn}', file=sf)
                print(f'', file=sf)
                print(_sfn_text, file=sf)
                print(f'', file=sf)

            else:
                _sfn = _get_sample_fn(fn)
                if _sfn is None:
                    continue

                try:
                    _sfn_text = _sfn(
                        repo_id=repo_id,
                        repo_type=repo_type,
                        filename=fn,
                        revision=revision,
                        hf_token=hf_token,
                    )
                except:
                    logging.exception(f'Sample skipped for file {fn!r}...')
                    continue

                print(f'## {fn}', file=sf)
                print(f'', file=sf)
                print(_sfn_text, file=sf)
                print(f'', file=sf)

        prompt = sf.getvalue()
        return prompt


def ask_llm_for_hf_repo_info(repo_id: str, repo_type: RepoTypeTyping = 'dataset', revision: str = 'main',
                             hf_token: Optional[str] = None, max_retries: int = 5, max_sample_count: int = 15):
    _SYSTEM_PROMPT = textwrap.dedent(f"""
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
        "is_ready_to_view": {{
            "yes": true/false,
            "reason": "reason why it is (not) ready, if not ready what should we do",
        }},
        "is_clear_enough": {{
            "yes": true/false,
            "reason": "reason why it is (not) clear, if not clear enough what should we do",
        }}
    }}
    ```

    ## Field Guidelines

    **repo_type**: Determine if this is a "model", "dataset", or "space" repository.

    **task_types**:
    - For model repositories: Identify the specific ML tasks this model can perform, must be accurate, do not contain those task types not quite sure enough due to the given information 
    - For dataset repositories: Identify ALL possible ML tasks this dataset can be used for, you need to analyze and make your own judgment, not do not contain those impossible ones (e.g. for datasets without mask plz do not given me image-segmentation)
    - For space repositories: Identify the tasks demonstrated or supported, must be accurate
    - Available options include: {", ".join(_TASK_TYPES)}, and others as appropriate.

    **libraries**:
    - Identify the key libraries, frameworks, and tools used or supported.
    - For dataset, you should only use those libraries which explicitly mentioned or supported due to the README for some data. 
    - Common options include: {", ".join(ML_FRAMEWORKS)}, and others as appropriate.

    **modality**: Identify all data modalities involved. Options include: Text, Image, Audio, Video, Tabular, Time Series, Graph, 3D, Multi-Modal, and others as appropriate.

    **abstract**: Write a comprehensive 150-200 word summary following academic paper abstract conventions but adapted for open-source repositories:

    - **For model repositories**:
      - Start with the problem statement and motivation
      - Describe the technical approach, architecture innovations, and key methodological contributions
      - Present quantitative results, performance metrics, and comparisons with existing methods when available
      - Highlight the model's capabilities, supported tasks, and technical specifications
      - Discuss the significance for both academic research and industrial applications
      - Mention training data scale, computational requirements, or efficiency improvements if relevant

    - **For dataset repositories**:
      - Begin with the research problem or application domain this dataset addresses
      - Describe the data collection methodology, curation process, and quality assurance measures
      - Provide quantitative details: dataset size, number of samples, annotation types, and coverage statistics
      - Explain the dataset's unique characteristics, annotations, and technical format specifications
      - Discuss potential research applications and how it advances the field
      - Address limitations, ethical considerations, or novel aspects that distinguish it from existing datasets

    - **For space repositories**:
      - Introduce the demonstrated application or interactive system's purpose
      - Describe the underlying technical implementation, model integration, and system architecture
      - Explain the user interface design, interaction capabilities, and supported functionalities
      - Present the technical stack, performance characteristics, and scalability features
      - Discuss the educational, research, or practical value for the community
      - Highlight how it bridges research and practical applications or enables new use cases

    **bio**: Write ONE clear sentence that describes what this repository is used for.

    **keywords**: Provide 3-5 relevant keywords that complement the abstract and help categorize the repository, should be matched with abstract part. All keywords should be capitalized and titleized, like 'WebDataset', 'Image Classification'.

    **is_ready_to_view**: Set to `true` if the repository has a comprehensive README that clearly guides users on how to use it. Set to `false` if the README is missing, incomplete, or unclear. And attention that 'gated' is not a reason for not ready enough, the user is able to see the README and folder directory even when HF_TOKEN is not provided.

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
    """).strip()

    client = get_hf_client(hf_token=hf_token)
    repo_info = client.repo_info(
        repo_id=repo_id,
        repo_type=repo_type,
        expand=['author', 'cardData', 'createdAt', 'disabled', 'downloads',
                'downloadsAllTime', 'gated', 'lastModified', 'likes', 'private', 'tags',
                'trendingScore', 'xetEnabled', 'usedStorage'] if repo_type != 'space' else
        ['author', 'cardData', 'createdAt', 'disabled', 'lastModified', 'likes', 'private', 'tags',
         'trendingScore', 'xetEnabled', 'usedStorage'],
    )

    cnt = 0
    while True:
        try:
            text = ask_llm(
                prompts=[
                    {
                        "role": "system",
                        "content": _SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": get_hf_repo_abstract_prompt(
                            repo_id=repo_id,
                            repo_type=repo_type,
                            revision=revision,
                            hf_token=hf_token,
                            max_retries=max_retries,
                            max_sample_count=max_sample_count,
                        ),
                    }
                ],
                # model_name='deepseek-reasoner',
                model_name='deepseek-chat',
            )
            return {
                **parse_json_from_llm_output(text),
                "repo_id": repo_id,
                "repo_type": repo_type,
                "downloads": repo_info.downloads_all_time if repo_type != 'space' else None,
                "likes": repo_info.likes,
                "created_at": time.time(),
            }
        except:
            cnt += 1
            if cnt > max_retries:
                raise
            logging.exception(f'Error on parsing ({cnt}/{max_retries}) ...')
