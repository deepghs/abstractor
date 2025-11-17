import os
import textwrap
from typing import Optional

from ditk import logging
from hbutils.system import TemporaryDirectory
from hfutils.operate import upload_directory_as_directory
from hfutils.operate.base import RepoTypeTyping

from abstractor.openai import ask_llm
from abstractor.repository import get_hf_repo_abstract_prompt


def sync(repo_id: str, repo_type: RepoTypeTyping = 'dataset', revision: str = 'main',
         hf_token: Optional[str] = None, max_retries: int = 5, max_sample_count: int = 15,
         extra_text: str = ''):
    _SYSTEM_PROMPT = textwrap.dedent("""
You are an expert AI assistant specialized in creating professional README.md files for Hugging Face repositories. Follow these comprehensive guidelines to generate perfect documentation:

#### 1. Repository Type Identification
- **Mandatory**: User will specify if it's a `model` or `dataset` repository
- If unspecified, infer from file structure:
  - Model indicators: `pytorch_model.bin`, `model.safetensors`, `config.json`
  - Dataset indicators: `dataset_info.json`, `data/*.arrow`, `data/*.parquet`

#### 2. Metadata Block Requirements
**Position**: Must be the first element in README, wrapped in `---`

**Model Repository Metadata**:
```yaml
pipeline_tag: (REQUIRED) # Official task type
  - Options: text-classification, text-generation, token-classification, 
             question-answering, fill-mask, automatic-speech-recognition,
             image-classification, object-detection, text-to-image
  - Format: Single string value

license: (REQUIRED) # SPDX identifier
  - Options: apache-2.0, mit, bsd-3-clause, gpl-3.0, cc-by-4.0, 
             cc-by-sa-4.0, openrail, creativeml-openrail-m
  - Default: apache-2.0 (if no original license found)

tags: (STRONGLY RECOMMENDED)
  - Structure: Array of strings
  - Minimum: 3 tags (task type + framework + architecture)
  - Framework options: pytorch, tensorflow, jax, onnx, scikit-learn
  - Architecture options: bert, roberta, gpt2, t5, resnet, yolos

# Optional fields (include if information available)
datasets: 
  - Format: [glue, squad, username/custom-dataset]
metrics:
  - Format: [accuracy, f1, bleu, rouge, wer]
library_name: transformers  # or diffusers, sentence-transformers
base_model: bert-base-uncased
widget:
  - text: "Example input"
    example_title: "Sample Demonstration"
```

**Dataset Repository Metadata**:
```yaml
task_categories: (REQUIRED)
  - Format: [text-classification, question-answering, translation]
language: (REQUIRED) # ISO 639-1 codes
  - Format: [en, zh, fr, de, es, multilingual]
license: (REQUIRED) 
  - Default: cc-by-4.0 (if no original license found)

# Recommended fields
size_categories: 
  - Options: n<1K, 1K<n<10K, 10K<n<100K, 100K<n<1M, 1M<n<10M, n>10M
annotations_creators: 
  - Options: crowdsourced, expert-generated, machine-generated, found
language_creators: 
  - Options: same as annotations_creators
multilinguality: 
  - Options: monolingual, multilingual
source_datasets: 
  - Format: [original, extended|username/dataset]

# Optional fields
task_ids: [named-entity-recognition, sentiment-analysis]
configs: [en_base, zh_simplified]
version: 1.0.0
```

#### 3. Content Structure Template
````markdown
---
# METADATA BLOCK HERE
---

# [Main Title: Concise and Descriptive]

## Summary
[200+ word description with 3-5 **bolded keywords**]
- First paragraph: Core functionality and key innovation
- Second paragraph: Technical approach and methodology
- Third paragraph: Performance highlights and use cases
- Fourth paragraph: Dataset/model characteristics
- Keywords: **Keyword1**, **Keyword2**, **Keyword3**

## Usage
```bash
# Installation
pip install [required packages]
```
```python
# Example Code
[Complete runnable example with imports, loading, and execution]
```

## Original Content
[EXACT preservation of original tables, code blocks, and diagrams]

# [Original Section Headers]
[Reorganized original content with preserved wording]

# Citation
```bibtex
[Complete BibTeX entry - preserve original or generate if available]
```
````

#### 4. Citation Section Rules
1. **Position**: Always place at the **end** of the README
2. **Content Determination**:
   - **Non-Original Work** (derivative models/datasets):
     - Point to original authors and source
     - Include full original citation details
     - Add note: "This repository provides an implementation/adaptation of the original work"
   - **Original Work** (novel contribution):
     - Cite this repository as the primary source
     - Include comprehensive details about the work
3. **Format Requirements**:
   - Wrap in triple-backtick code block with `bibtex` language identifier
   - Follow standard BibTeX format with complete metadata

4. **Citation Templates**:
```bibtex
# For ORIGINAL WORK (this repository is the source)
@misc{repository_identifier,
  title        = {{Full Repository Title}},
  author       = {Author1 and Author2 and ...},
  howpublished = {\\url{https://huggingface.co/username/repo}},
  year         = {2023},
  note         = {Summary of key contributions: [1-2 sentence abstract]},
  abstract     = {Full 200+ word abstract from the Summary section},
  keywords     = {Keyword1, Keyword2, Keyword3}
}

# For NON-ORIGINAL WORK (pointing to original source)
@inproceedings{original_paper,
  title     = {Original Paper Title},
  author    = {Original Author1 and Original Author2},
  booktitle = {Conference/Journal Name},
  year      = {2020},
  url       = {https://arxiv.org/abs/xxxx.xxxxx}
}
@misc{original_repository,
  title        = {Original Repository Title},
  author       = {Original Maintainer},
  howpublished = {\\url{https://huggingface.co/original/repo}},
  year         = {2022},
  note         = {This repository provides an implementation/adaptation of the original work}
}
```

5. **Metadata Requirements**:
   - Must include:
     - `title`: Full repository name/title
     - `author`: All contributors (format: "Name1 and Name2 and ...")
     - `howpublished`: Full Hugging Face repository URL
     - `year`: Creation/publication year
     - `note`: Brief 1-2 sentence summary of key contributions
     - `abstract`: Full abstract from the Summary section
     - `keywords`: 3-5 keywords from the Summary section
   - Optional but recommended:
     - `doi`: Digital Object Identifier if available
     - `version`: Model/dataset version

6. **Originality Determination**:
   - Considered **Original Work** if:
     - Novel architecture/methodology introduced
     - New dataset created (not derived from existing)
     - Significant original contributions beyond fine-tuning
   - Considered **Non-Original** if:
     - Direct fine-tune of existing model
     - Minor adaptation of existing work
     - Derived from existing dataset with minimal changes

7. **Information Sourcing**:
   - Author names: From repository owners/contributors
   - Year: From creation date in repository history
   - Abstract: Use the generated Summary section content
   - Keywords: Use the bolded keywords from Summary
   - Title: Use the main repository title

8. **Fallback for Missing Information**:
```bibtex
@misc{repository,
  title        = {{Repository Title}},
  author       = {Repository Contributors},
  howpublished = {\\url{https://huggingface.co/username/repo}},
  year         = {2023},
  note         = {[Brief description from metadata or summary]}
}
```

This revised approach ensures proper attribution while providing rich, reusable citation metadata that:
1. Gives appropriate credit for original work
2. Provides comprehensive information for academic citation
3. Maintains direct connection to the repository
4. Includes all necessary elements for LaTeX/BibTeX users
5. Clearly distinguishes between original and derivative works

#### 5. Critical Preservation Rules
1. **Exact Copy Requirements**:
   - Tables: Preserve every character, space, and alignment
   - Code blocks: Maintain original formatting and comments
   - Mathematical notation: Keep all LaTeX expressions ($...$ and $$...$$) unchanged
   - Hyperlinks: Preserve all URL formats and anchor texts

2. **Content Organization**:
   - Retain all original section headers
   - Maintain original bullet points and numbering
   - Preserve all image references: `![Alt Text](url)`
   - Keep all citations and references intact

3. **License Handling**:
   - Body text: Only include license information if present in original README
   - Metadata: Always include license field (use defaults if missing)

4. Use dghs-imgutils/dghs-realutils library directly if possible, do not just repeat the code from them.

5. The provided source code in extra content part may contain rst format docs, if you use those content you should make sure all the content is transformed to markdown fomat. e.g. `xxx <yyy>`_ to [xxxx](yyyy), etc.

#### 6. Special Case Handling
- **No original README**:
  - Generate comprehensive summary from file structure and data samples
  - Create usage section from downstream library documentation
  - Include "Original Content" section with repository file tree

- **Multiple languages**:
  - Add language-specific tags (zh, ja, ko)
  - Include multilingual examples in usage section
  - Preserve all original language content without translation

- **Conflicting information**:
  - Prioritize original README content over user descriptions
  - Preserve original technical specifications verbatim
  - Add metadata comments for potential inconsistencies

#### 7. Validation Checks
Before output, verify:
- Metadata block is valid YAML with required fields
- Summary contains ≥200 words and 3-5 bolded keywords
- All original tables/code blocks are preserved exactly
- License field exists in metadata (even if missing in body)
- Citation section is properly formatted BibTeX
- No external commentary or markdown wrappers added

**Output Format**: Raw README.md content only - no prefixes, suffixes, or explanations. Ready for direct use on Hugging Face Hub.
    """).strip()

    cnt = 0
    text = None
    while cnt < max_retries:
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
                        ) + f'\n\n## Extra Content\n\n{textwrap.dedent(extra_text).strip()}',
                    }
                ],
                # model_name='deepseek-reasoner',
                model_name='deepseek-chat',
            )

        except:
            cnt += 1
            if cnt > max_retries:
                raise
            logging.exception(f'Error on parsing ({cnt}/{max_retries}) ...')
        else:
            break

    if text:
        with TemporaryDirectory() as upload_dir:
            with open(os.path.join(upload_dir, 'README.md'), 'w') as f:
                print(text, file=f)

            upload_directory_as_directory(
                repo_id=repo_id,
                repo_type=repo_type,
                local_directory=upload_dir,
                path_in_repo='.',
                message=f'Auto-update README.md via abstractor',
            )


if __name__ == '__main__':
    logging.try_init_root(level=logging.INFO)
    sync(
        repo_id='deepghs/image_restoration',
        repo_type='model',
        extra_text=('''
Source code from library `dghs-imgutils`, you can install it with `pip install dghs-imgutils`


Code 'imgutils/restore/nafnet.py':

"""
Overview:
    Restore the images using `NafNet <https://github.com/megvii-research/NAFNet>`_.

    .. image:: nafnet_demo.plot.py.svg
        :align: center

    This is an overall benchmark of all the NafNet models:

    .. image:: nafnet_benchmark.plot.py.svg
        :align: center

    .. warning::
        Currently, we've identified a significant issue with NafNet when images contain gaussian noise.
        To ensure your code functions correctly, please ensure the credibility of
        your image source or preprocess them using SCUNet.

    .. note::
        New in version v0.4.4, **images with alpha channel supported**.

        If you use an image with alpha channel (e.g. RGBA images),
        it will return a RGBA image, otherwise return RGG image.
"""
from typing import Literal

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from ..data import ImageTyping
from ..generic import ImageEnhancer
from ..utils import open_onnx_model, area_batch_run, ts_lru_cache

NafNetModelTyping = Literal['REDS', 'GoPro', 'SIDD']


@ts_lru_cache()
def _open_nafnet_model(model: NafNetModelTyping):
    """
    Open the NAFNet model for image restoration.

    :param model: The NAFNet model type ('REDS', 'GoPro', 'SIDD').
    :type model: NafNetModelTyping
    :return: The NAFNet ONNX model.
    """
    return open_onnx_model(hf_hub_download(
        f'deepghs/image_restoration',
        f'NAFNet-{model}-width64.onnx',
    ))


class _Enhancer(ImageEnhancer):
    def __init__(self, model: NafNetModelTyping = 'REDS', tile_size: int = 256, tile_overlap: int = 16,
                 batch_size: int = 4, silent: bool = False):
        self.model = model
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.batch_size = batch_size
        self.silent = silent

    def _process_rgb(self, rgb_array: np.ndarray):
        input_ = rgb_array[None, ...]

        def _method(ix):
            ox, = _open_nafnet_model(self.model).run(['output'], {'input': ix})
            return ox

        output_ = area_batch_run(
            input_, _method,
            tile_size=self.tile_size, tile_overlap=self.tile_overlap, batch_size=self.batch_size,
            silent=self.silent, process_title='NafNet Restore',
        )
        output_ = np.clip(output_, a_min=0.0, a_max=1.0)
        return output_[0]


@ts_lru_cache()
def _get_enhancer(model: NafNetModelTyping = 'REDS', tile_size: int = 256, tile_overlap: int = 16,
                  batch_size: int = 4, silent: bool = False) -> _Enhancer:
    return _Enhancer(model, tile_size, tile_overlap, batch_size, silent)


def restore_with_nafnet(image: ImageTyping, model: NafNetModelTyping = 'REDS',
                        tile_size: int = 256, tile_overlap: int = 16, batch_size: int = 4,
                        silent: bool = False) -> Image.Image:
    """
    Restore an image using the NAFNet model.

    :param image: The input image.
    :type image: ImageTyping
    :param model: The NAFNet model type ('REDS', 'GoPro', 'SIDD'). Default is 'REDS'.
    :type model: NafNetModelTyping
    :param tile_size: The size of processing tiles. Default is 256.
    :type tile_size: int
    :param tile_overlap: The overlap between tiles. Default is 16.
    :type tile_overlap: int
    :param batch_size: The batch size of inference. Default is 4.
    :type batch_size: int
    :param silent: If True, the progress will not be displayed. Default is False.
    :type silent: bool
    :return: The restored image.
    :rtype: Image.Image
    """
    return _get_enhancer(model, tile_size, tile_overlap, batch_size, silent).process(image)


Code 'imgutils/restore/scunet.py':

"""
Overview:
    Restore the images using `SCUNet <https://github.com/cszn/SCUNet>`_.

    .. image:: scunet_demo.plot.py.svg
        :align: center

    This is an overall benchmark of all the SCUNet models:

    .. image:: scunet_benchmark.plot.py.svg
        :align: center

    .. note::
        New in version v0.4.4, **images with alpha channel supported**.

        If you use an image with alpha channel (e.g. RGBA images),
        it will return a RGBA image, otherwise return RGG image.
"""
from typing import Literal

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from ..data import ImageTyping
from ..generic import ImageEnhancer
from ..utils import open_onnx_model, area_batch_run, ts_lru_cache

SCUNetModelTyping = Literal['GAN', 'PSNR']


@ts_lru_cache()
def _open_scunet_model(model: SCUNetModelTyping):
    """
    Open the SCUNet model for image restoration.

    :param model: The SCUNet model type ('GAN', 'PSNR').
    :type model: SCUNetModelTyping
    :return: The SCUNet ONNX model.
    """
    return open_onnx_model(hf_hub_download(
        f'deepghs/image_restoration',
        f'SCUNet-{model}.onnx'
    ))


class _Enhancer(ImageEnhancer):
    def __init__(self, model: SCUNetModelTyping = 'GAN', tile_size: int = 128, tile_overlap: int = 16,
                 batch_size: int = 4, silent: bool = False):
        self.model = model
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.batch_size = batch_size
        self.silent = silent

    def _process_rgb(self, rgb_array: np.ndarray):
        input_ = rgb_array[None, ...]

        def _method(ix):
            ox, = _open_scunet_model(self.model).run(['output'], {'input': ix})
            return ox

        output_ = area_batch_run(
            input_, _method,
            tile_size=self.tile_size, tile_overlap=self.tile_overlap, batch_size=self.batch_size,
            silent=self.silent, process_title='SCUNet Restore',
        )
        output_ = np.clip(output_, a_min=0.0, a_max=1.0)
        return output_[0]


@ts_lru_cache()
def _get_enhancer(model: SCUNetModelTyping = 'GAN', tile_size: int = 128, tile_overlap: int = 16,
                  batch_size: int = 4, silent: bool = False) -> _Enhancer:
    return _Enhancer(model, tile_size, tile_overlap, batch_size, silent)


def restore_with_scunet(image: ImageTyping, model: SCUNetModelTyping = 'GAN',
                        tile_size: int = 128, tile_overlap: int = 16, batch_size: int = 4,
                        silent: bool = False) -> Image.Image:
    """
    Restore an image using the SCUNet model.

    :param image: The input image.
    :type image: ImageTyping
    :param model: The SCUNet model type ('GAN', 'PSNR'). Default is 'GAN'.
    :type model: SCUNetModelTyping
    :param tile_size: The size of processing tiles. Default is 128.
    :type tile_size: int
    :param tile_overlap: The overlap between tiles. Default is 16.
    :type tile_overlap: int
    :param batch_size: The batch size of inference. Default is 4.
    :type batch_size: int
    :param silent: If True, the progress will not be displayed. Default is False.
    :type silent: bool
    :return: The restored image.
    :rtype: Image.Image
    """
    return _get_enhancer(model, tile_size, tile_overlap, batch_size, silent).process(image)


Code 'imgutils/restore/__init__.py':

"""
Overview:
    Utilities for restoring images, which may be jpeg, blurry or noisy.

    The following models are used:

    * `NafNet <https://github.com/megvii-research/NAFNet>`_
    * `SCUNet <https://github.com/cszn/SCUNet>`_

    .. image:: restore_demo.plot.py.svg
        :align: center

"""
from .adversarial import remove_adversarial_noise
from .nafnet import restore_with_nafnet
from .scunet import restore_with_scunet



        ''')
    )
