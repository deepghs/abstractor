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
        repo_id='deepghs/ml-danbooru-onnx',
        repo_type='model',
        extra_text=('''
Source code from library `dghs-imgutils`, you can install it with `pip install dghs-imgutils`

And this guy 7eu7d7 is a member of deepghs as well, so this can be treated as original-created by deepghs as well.

Code 'imgutils/tagging/mldanbooru.py':
        
"""
Overview:
    Tagging utils based on ML-danbooru which is provided by 7eu7d7. The code is here:
    `7eu7d7/ML-Danbooru <https://github.com/7eu7d7/ML-Danbooru>`_ .
"""
from typing import Tuple, List

import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download

from .overlap import drop_overlap_tags
from ..data import load_image, ImageTyping
from ..utils import open_onnx_model, ts_lru_cache


@ts_lru_cache()
def _open_mldanbooru_model():
    return open_onnx_model(hf_hub_download('deepghs/ml-danbooru-onnx', 'ml_caformer_m36_dec-5-97527.onnx'))


def _resize_align(image: Image.Image, size: int, keep_ratio: float = True, align: int = 4) -> Image.Image:
    if not keep_ratio:
        target_size = (size, size)
    else:
        min_edge = min(image.size)
        target_size = (
            int(image.size[0] / min_edge * size),
            int(image.size[1] / min_edge * size),
        )

    target_size = (
        (target_size[0] // align) * align,
        (target_size[1] // align) * align,
    )

    return image.resize(target_size, resample=Image.BILINEAR)


def _to_tensor(image: Image.Image):
    # noinspection PyTypeChecker
    img: np.ndarray = np.array(image, dtype=np.uint8, copy=True)
    img = img.reshape((image.size[1], image.size[0], len(image.getbands())))

    # put it from HWC to CHW format
    img = img.transpose((2, 0, 1))
    return img.astype(np.float32) / 255


@ts_lru_cache()
def _get_mldanbooru_labels(use_real_name: bool = False) -> Tuple[List[str], List[int], List[int]]:
    path = hf_hub_download('deepghs/imgutils-models', 'mldanbooru/mldanbooru_tags.csv')
    df = pd.read_csv(path)

    return df["name"].tolist() if not use_real_name else df['real_name'].tolist()


def get_mldanbooru_tags(image: ImageTyping, use_real_name: bool = False,
                        threshold: float = 0.7, size: int = 448, keep_ratio: bool = False,
                        drop_overlap: bool = False):
    """
    Overview:
        Tagging image with ML-Danbooru, similar to
        `deepghs/ml-danbooru-demo <https://huggingface.co/spaces/deepghs/ml-danbooru-demo>`_.

    :param image: Image to tagging.
    :param use_real_name: Use real name on danbooru. Due to the renaming and redirection of many tags
        on the Danbooru website after the training of ``deepdanbooru``,
        it may be necessary to use the latest tag names in some application scenarios.
        The default value of ``False`` indicates the use of the original tag names.
    :param threshold: Threshold for tags, default is ``0.7``.
    :param size: Size when passing the resized image into model, default is ``448``.
    :param keep_ratio: Keep the original ratio between height and width when passing the image into
        model, default is ``False``.
    :param drop_overlap: Drop overlap tags or not, default is ``False``.

    Example:
        Here are some images for example

        .. image:: tagging_demo.plot.py.svg
           :align: center

        >>> import os
        >>> from imgutils.tagging import get_mldanbooru_tags
        >>>
        >>> get_mldanbooru_tags('skadi.jpg')
        {'1girl': 0.9999984502792358, 'long_hair': 0.9999946355819702, 'red_eyes': 0.9994951486587524, 'navel': 0.998144268989563, 'breasts': 0.9978417158126831, 'solo': 0.9941409230232239, 'shorts': 0.9799384474754333, 'gloves': 0.979142427444458, 'very_long_hair': 0.961823582649231, 'looking_at_viewer': 0.961323618888855, 'silver_hair': 0.9490893483161926, 'large_breasts': 0.9450850486755371, 'midriff': 0.9425153136253357, 'sweat': 0.9409335255622864, 'thighs': 0.9319437146186829, 'crop_top': 0.9265308976173401, 'baseball_bat': 0.9259042143821716, 'sky': 0.922250509262085, 'holding': 0.9199565052986145, 'outdoors': 0.9175475835800171, 'day': 0.9102761745452881, 'black_gloves': 0.9076938629150391, 'stomach': 0.9052775502204895, 'shirt': 0.8938589692115784, 'cowboy_shot': 0.8894285559654236, 'bangs': 0.8891903162002563, 'blue_sky': 0.8845980763435364, 'parted_lips': 0.8842408061027527, 'hair_between_eyes': 0.8659475445747375, 'sportswear': 0.862621009349823, 'no_headwear': 0.8616052865982056, 'cloud': 0.8562789559364319, 'short_shorts': 0.8555729389190674, 'no_hat': 0.8533340096473694, 'black_shorts': 0.8477485775947571, 'short_sleeves': 0.8430152535438538, 'low-tied_long_hair': 0.8340626955032349, 'crop_top_overhang': 0.8266023397445679, 'holding_baseball_bat': 0.8222048282623291, 'standing': 0.8202669620513916, 'black_shirt': 0.8061150312423706, 'ass_visible_through_thighs': 0.7803354859352112, 'thigh_gap': 0.7789446711540222, 'arms_up': 0.7052110433578491}
        >>>
        >>> get_mldanbooru_tags('hutao.jpg')
        {'1girl': 0.9999866485595703, 'skirt': 0.997043788433075, 'tongue': 0.9969649910926819, 'hair_ornament': 0.9957101345062256, 'tongue_out': 0.9928386807441711, 'flower': 0.9886980056762695, 'twintails': 0.9864778518676758, 'ghost': 0.9769423007965088, 'hair_flower': 0.9747489094734192, 'bag': 0.9736957550048828, 'long_hair': 0.9388670325279236, 'backpack': 0.9356311559677124, 'brown_hair': 0.91000896692276, 'cardigan': 0.8955123424530029, 'red_eyes': 0.8910233378410339, 'plaid': 0.8904104828834534, 'looking_at_viewer': 0.8881211280822754, 'school_uniform': 0.8876776695251465, 'outdoors': 0.8864808678627014, 'jacket': 0.8810517191886902, 'plaid_skirt': 0.8798807263374329, 'ahoge': 0.8765745162963867, 'pleated_skirt': 0.8737136125564575, 'nail_polish': 0.8650439381599426, 'solo': 0.8613706827163696, 'blue_cardigan': 0.8571277260780334, 'bangs': 0.8333670496940613, 'very_long_hair': 0.8160212635993958, 'eyebrows_visible_through_hair': 0.8122442364692688, 'hairclip': 0.8091571927070618, 'red_nails': 0.8082079887390137, ':p': 0.8048468232154846, 'long_sleeves': 0.8042327165603638, 'shirt': 0.7984272241592407, 'blazer': 0.794708251953125, 'ribbon': 0.78981614112854, 'hair_ribbon': 0.7892146110534668, 'star-shaped_pupils': 0.7867060899734497, 'gradient_hair': 0.786359965801239, 'white_shirt': 0.7790888547897339, 'brown_skirt': 0.7760675549507141, 'symbol-shaped_pupils': 0.774523913860321, 'smile': 0.7721588015556335, 'hair_between_eyes': 0.7697228789329529, 'cowboy_shot': 0.755959689617157, 'multicolored_hair': 0.7477189898490906, 'blush': 0.7476690411567688, 'railing': 0.7476617693901062, 'blue_jacket': 0.7458406090736389, 'sleeves_past_wrists': 0.741143524646759, 'day': 0.7364678978919983, 'collared_shirt': 0.7193643450737, 'red_neckwear': 0.7108616828918457, 'flower-shaped_pupils': 0.7086325287818909, 'miniskirt': 0.7055293321609497, 'holding': 0.7039415836334229, 'open_clothes': 0.7018357515335083}

    .. note::
        ML-Danbooru only contains generic tags, so the return value will not be splitted like that in
        :func:`imgutils.tagging.deepdanbooru.get_deepdanbooru_tags` or
        :func:`imgutils.tagging.wd14.get_wd14_tags`.
    """
    image = load_image(image, mode='RGB')
    real_input = _to_tensor(_resize_align(image, size, keep_ratio))
    real_input = real_input.reshape(1, *real_input.shape)

    model = _open_mldanbooru_model()
    native_output, = model.run(['output'], {'input': real_input})

    output = (1 / (1 + np.exp(-native_output))).reshape(-1)
    tags = _get_mldanbooru_labels(use_real_name)
    pairs = sorted([(tags[i], ratio) for i, ratio in enumerate(output)], key=lambda x: (-x[1], x[0]))

    general_tags = {tag: float(ratio) for tag, ratio in pairs if ratio >= threshold}
    if drop_overlap:
        general_tags = drop_overlap_tags(general_tags)
    return general_tags

Code 'imgutils/tagging/__init__.py':

"""
Overview:
    Get tags for anime images.

    This is an overall benchmark of all the danbooru models:

    .. image:: tagging_benchmark.plot.py.svg
        :align: center

"""
from .blacklist import is_blacklisted, drop_blacklisted_tags
from .camie import get_camie_tags, convert_camie_emb_to_prediction
from .character import is_basic_character_tag, drop_basic_character_tags
from .deepdanbooru import get_deepdanbooru_tags
from .deepgelbooru import get_deepgelbooru_tags
from .format import tags_to_text, add_underline, remove_underline
from .match import tag_match_suffix, tag_match_prefix, tag_match_full
from .mldanbooru import get_mldanbooru_tags
from .order import sort_tags
from .overlap import drop_overlap_tags
from .pixai import get_pixai_tags
from .wd14 import get_wd14_tags, convert_wd14_emb_to_prediction, denormalize_wd14_emb


        ''')
    )
