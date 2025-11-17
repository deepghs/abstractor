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
        repo_id='deepghs/paddleocr',
        repo_type='model',
        extra_text=('''
Source code from library `dghs-imgutils`, you can install it with `pip install dghs-imgutils`

Code 'imgutils/ocr/detect.py':
        
from typing import List

import cv2
import numpy as np
import pyclipper
from huggingface_hub import hf_hub_download, HfFileSystem
from shapely import Polygon

from ..data import ImageTyping, load_image
from ..utils import open_onnx_model, ts_lru_cache

_MIN_SIZE = 3
_HF_CLIENT = HfFileSystem()
_REPOSITORY = 'deepghs/paddleocr'


@ts_lru_cache()
def _open_ocr_detection_model(model):
    return open_onnx_model(hf_hub_download(
        _REPOSITORY,
        f'det/{model}/model.onnx',
    ))


def _box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    # noinspection PyTypeChecker
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def _unclip(box, unclip_ratio):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


def _get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0

    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [
        points[index_1], points[index_2], points[index_3], points[index_4]
    ]
    return box, min(bounding_box[1])


def _boxes_from_bitmap(pred, _bitmap, dest_width, dest_height,
                       box_threshold=0.7, max_candidates=1000, unclip_ratio=2.0):
    bitmap = _bitmap
    height, width = bitmap.shape

    outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(outs) == 3:
        img, contours, _ = outs[0], outs[1], outs[2]
    elif len(outs) == 2:
        contours, _ = outs[0], outs[1]

    # noinspection PyUnboundLocalVariable
    num_contours = min(len(contours), max_candidates)

    boxes = []
    scores = []
    for index in range(num_contours):
        contour = contours[index]
        points, sside = _get_mini_boxes(contour)
        if sside < _MIN_SIZE:
            continue
        points = np.array(points)
        score = _box_score_fast(pred, points.reshape(-1, 2))
        if box_threshold > score:
            continue

        box = _unclip(points, unclip_ratio).reshape(-1, 1, 2)
        box, sside = _get_mini_boxes(box)
        if sside < _MIN_SIZE + 2:
            continue
        box = np.array(box)

        box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
        boxes.append(box.astype("int32"))
        scores.append(score)
    return np.array(boxes, dtype="int32"), scores


def _normalize(data, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)):
    mean, std = np.asarray(mean), np.asarray(std)
    return (data - mean[None, :, None, None]) / std[None, :, None, None]


_ALIGN = 64


def _get_text_points(image: ImageTyping, model: str = 'ch_PP-OCRv4_det',
                     heat_threshold: float = 0.3, box_threshold: float = 0.7,
                     max_candidates: int = 1000, unclip_ratio: float = 2.0):
    origin_width, origin_height = width, height = image.size
    if width % _ALIGN != 0:
        width += (_ALIGN - width % _ALIGN)
    if height % _ALIGN != 0:
        height += (_ALIGN - height % _ALIGN)

    input_ = np.array(image).transpose((2, 0, 1)).astype(np.float32) / 255.0
    # noinspection PyTypeChecker
    input_ = np.pad(input_[None, ...], ((0, 0), (0, 0), (0, height - origin_height), (0, width - origin_width)))

    _ort_session = _open_ocr_detection_model(model)

    input_ = _normalize(input_).astype(np.float32)
    _input_name = _ort_session.get_inputs()[0].name
    _output_name = _ort_session.get_outputs()[0].name
    output_, = _ort_session.run([_output_name], {_input_name: input_})
    heatmap = output_[0][0]
    heatmap = heatmap[:origin_height, :origin_width]

    retval = []
    for points, score in zip(*_boxes_from_bitmap(
            heatmap, heatmap >= heat_threshold, origin_width, origin_height,
            box_threshold, max_candidates, unclip_ratio,
    )):
        retval.append((points, score))
    return retval


def _detect_text(image: ImageTyping, model: str = 'ch_PP-OCRv4_det',
                 heat_threshold: float = 0.3, box_threshold: float = 0.7,
                 max_candidates: int = 1000, unclip_ratio: float = 2.0):
    image = load_image(image, force_background='white', mode='RGB')
    retval = []
    for points, score in _get_text_points(image, model, heat_threshold, box_threshold, max_candidates, unclip_ratio):
        x0, y0 = points[:, 0].min(), points[:, 1].min()
        x1, y1 = points[:, 0].max(), points[:, 1].max()
        retval.append(((x0.item(), y0.item(), x1.item(), y1.item()), 'text', score))

    return retval


@ts_lru_cache()
def _list_det_models() -> List[str]:
    retval = []
    repo_segment_cnt = len(_REPOSITORY.split('/'))
    for item in _HF_CLIENT.glob(f'{_REPOSITORY}/det/*/model.onnx'):
        retval.append(item.split('/')[repo_segment_cnt:][1])
    return retval

Code 'imgutils/ocr/entry.py'

from typing import List, Tuple

from .detect import _detect_text, _list_det_models
from .recognize import _text_recognize, _list_rec_models
from ..data import ImageTyping, load_image

_DEFAULT_DET_MODEL = 'ch_PP-OCRv4_det'
_DEFAULT_REC_MODEL = 'ch_PP-OCRv4_rec'


def list_det_models() -> List[str]:
    """
    List available text detection models for OCR.

    :return: A list of available text detection model names.
    :rtype: List[str]

    Examples::
        >>> from imgutils.ocr import list_det_models
        >>>
        >>> list_det_models()
        ['ch_PP-OCRv2_det',
         'ch_PP-OCRv3_det',
         'ch_PP-OCRv4_det',
         'ch_PP-OCRv4_server_det',
         'ch_ppocr_mobile_slim_v2.0_det',
         'ch_ppocr_mobile_v2.0_det',
         'ch_ppocr_server_v2.0_det',
         'en_PP-OCRv3_det']
    """
    return _list_det_models()


def list_rec_models() -> List[str]:
    """
    List available text recognition models for OCR.

    :return: A list of available text recognition model names.
    :rtype: List[str]

    Examples::
        >>> from imgutils.ocr import list_rec_models
        >>>
        >>> list_rec_models()
        ['arabic_PP-OCRv3_rec',
         'ch_PP-OCRv2_rec',
         'ch_PP-OCRv3_rec',
         'ch_PP-OCRv4_rec',
         'ch_PP-OCRv4_server_rec',
         'ch_ppocr_mobile_v2.0_rec',
         'ch_ppocr_server_v2.0_rec',
         'chinese_cht_PP-OCRv3_rec',
         'cyrillic_PP-OCRv3_rec',
         'devanagari_PP-OCRv3_rec',
         'en_PP-OCRv3_rec',
         'en_PP-OCRv4_rec',
         'en_number_mobile_v2.0_rec',
         'japan_PP-OCRv3_rec',
         'ka_PP-OCRv3_rec',
         'korean_PP-OCRv3_rec',
         'latin_PP-OCRv3_rec',
         'ta_PP-OCRv3_rec',
         'te_PP-OCRv3_rec']
    """
    return _list_rec_models()


def detect_text_with_ocr(image: ImageTyping, model: str = _DEFAULT_DET_MODEL,
                         heat_threshold: float = 0.3, box_threshold: float = 0.7,
                         max_candidates: int = 1000, unclip_ratio: float = 2.0) \
        -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """
    Detect text in an image using an OCR model.

    :param image: The input image.
    :type image: ImageTyping
    :param model: The name of the text detection model.
    :type model: str, optional
    :param heat_threshold: The heat map threshold for text detection.
    :type heat_threshold: float, optional
    :param box_threshold: The box threshold for text detection.
    :type box_threshold: float, optional
    :param max_candidates: The maximum number of candidates to consider.
    :type max_candidates: int, optional
    :param unclip_ratio: The unclip ratio for text detection.
    :type unclip_ratio: float, optional
    :return: A list of detected text boxes, label (always ``text``), and their confidence scores.
    :rtype: List[Tuple[Tuple[int, int, int, int], str, float]]

    Examples::
        >>> from imgutils.ocr import detect_text_with_ocr
        >>>
        >>> detect_text_with_ocr('comic.jpg')
        [((742, 485, 809, 511), 'text', 0.9543377610144915),
         ((682, 98, 734, 124), 'text', 0.9309689495575223),
         ((716, 136, 836, 164), 'text', 0.9042856988923695),
         ((144, 455, 196, 485), 'text', 0.874083638387722),
         ((719, 455, 835, 488), 'text', 0.8628696346175078),
         ((124, 478, 214, 508), 'text', 0.848871771901487),
         ((1030, 557, 1184, 578), 'text', 0.8352495440618789),
         ((427, 129, 553, 154), 'text', 0.8249209443996619)]

    .. note::
        If you need to extract the actual text content, use the :func:`ocr` function.
    """
    retval = []
    for box, _, score in _detect_text(image, model, heat_threshold, box_threshold, max_candidates, unclip_ratio):
        retval.append((box, 'text', score))
    retval = sorted(retval, key=lambda x: x[2], reverse=True)
    return retval


def ocr(image: ImageTyping, detect_model: str = _DEFAULT_DET_MODEL,
        recognize_model: str = _DEFAULT_REC_MODEL, heat_threshold: float = 0.3, box_threshold: float = 0.7,
        max_candidates: int = 1000, unclip_ratio: float = 2.0, rotation_threshold: float = 1.5,
        is_remove_duplicate: bool = False):
    """
    Perform optical character recognition (OCR) on an image.

    :param image: The input image.
    :type image: ImageTyping
    :param detect_model: The name of the text detection model.
    :type detect_model: str, optional
    :param recognize_model: The name of the text recognition model.
    :type recognize_model: str, optional
    :param heat_threshold: The heat map threshold for text detection.
    :type heat_threshold: float, optional
    :param box_threshold: The box threshold for text detection.
    :type box_threshold: float, optional
    :param max_candidates: The maximum number of candidates to consider.
    :type max_candidates: int, optional
    :param unclip_ratio: The unclip ratio for text detection.
    :type unclip_ratio: float, optional
    :param rotation_threshold: The rotation threshold for text detection.
    :type rotation_threshold: float, optional
    :param is_remove_duplicate: Whether to remove duplicate text content.
    :type is_remove_duplicate: bool, optional
    :return: A list of detected text boxes, their corresponding text content, and their combined confidence scores.
    :rtype: List[Tuple[Tuple[int, int, int, int], str, float]]

    Examples::
        >>> from imgutils.ocr import ocr
        >>>
        >>> ocr('comic.jpg')
        [((742, 485, 809, 511), 'MOB.', 0.9356705927336156),
         ((716, 136, 836, 164), 'SHISHOU,', 0.8933000384412466),
         ((682, 98, 734, 124), 'BUT', 0.8730931912907247),
         ((144, 455, 196, 485), 'OH,', 0.8417627579351514),
         ((427, 129, 553, 154), 'A MIRROR.', 0.7366019454049503),
         ((1030, 557, 1184, 578), '(EL)  GATO IBERICO', 0.7271127306351021),
         ((719, 455, 835, 488), "THAt'S △", 0.701928390168364),
         ((124, 478, 214, 508), 'LOOK!', 0.6965972578194936)]

        By default, the text recognition model used is `ch_PP-OCRv4_rec`.
        This recognition model has good recognition capabilities for both Chinese and English.
        For unsupported text types, its recognition accuracy cannot be guaranteed, resulting in a lower score.
        **If you need recognition for other languages, please use :func:`list_rec_models` to
        view more available recognition models and choose the appropriate one for recognition.**

        >>> from imgutils.ocr import ocr
        >>>
        >>> # use default recognition model on japanese post
        >>> ocr('post_text.jpg')
        [
            ((319, 847, 561, 899), 'KanColle', 0.9130667787597329),
            ((552, 811, 791, 921), '1944', 0.8566762346615406),
            ((319, 820, 558, 850), 'Fleet  Girls Collection', 0.8100635458911772),
            ((235, 904, 855, 1009), '海', 0.6716076803280185),
            ((239, 768, 858, 808), 'I ·  tSu · ka ·  A· NO· u·  mI ·  de', 0.654507230718228),
            ((209, 507, 899, 811), '[', 0.2888084133529467)
        ]
        >>>
        >>> # use japanese model
        >>> ocr('post_text.jpg', recognize_model='japan_PP-OCRv3_rec')
        [
            ((319, 847, 561, 899), 'KanColle', 0.9230690942939336),
            ((552, 811, 791, 921), '1944', 0.8564870717047623),
            ((235, 904, 855, 1009), 'いつかあの海で', 0.8061289060358996),
            ((319, 820, 558, 850), 'Fleet   Girls  Collection', 0.8045396777081609),
            ((239, 768, 858, 808), 'I.TSU.KA・A・NO.U・MI.DE', 0.7311649382696896),
            ((209, 507, 899, 811), '「艦とれれ', 0.6648729016512889)
        ]

    """
    image = load_image(image)
    retval = []
    for (x0, y0, x1, y1), _, score in \
            _detect_text(image, detect_model, heat_threshold, box_threshold, max_candidates, unclip_ratio):
        width, height = x1 - x0, y1 - y0
        area = image.crop((x0, y0, x1, y1))
        if height >= width * rotation_threshold:
            area = area.rotate(90)

        text, rec_score = _text_recognize(area, recognize_model, is_remove_duplicate)
        retval.append(((x0, y0, x1, y1), text, score * rec_score))

    retval = sorted(retval, key=lambda x: x[2], reverse=True)
    return retval


Code 'imgutils/ocr/recognize.py'

from typing import List, Tuple

import numpy as np
from huggingface_hub import hf_hub_download, HfFileSystem

from ..data import ImageTyping, load_image
from ..utils import open_onnx_model, ts_lru_cache

_HF_CLIENT = HfFileSystem()
_REPOSITORY = 'deepghs/paddleocr'


@ts_lru_cache()
def _open_ocr_recognition_model(model):
    return open_onnx_model(hf_hub_download(
        _REPOSITORY,
        f'rec/{model}/model.onnx',
    ))


@ts_lru_cache()
def _open_ocr_recognition_dictionary(model) -> List[str]:
    with open(hf_hub_download(
            _REPOSITORY,
            f'rec/{model}/dict.txt',
    ), 'r', encoding='utf-8') as f:
        dict_ = [line.strip() for line in f]

    return ['<blank>', *dict_, ' ']


def _text_decode(text_index, model: str, text_prob=None, is_remove_duplicate=False):
    retval = []
    ignored_tokens = [0]
    batch_size = len(text_index)
    for batch_idx in range(batch_size):
        selection = np.ones(len(text_index[batch_idx]), dtype=bool)
        if is_remove_duplicate:
            selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
        for ignored_token in ignored_tokens:
            selection &= text_index[batch_idx] != ignored_token

        _dict = _open_ocr_recognition_dictionary(model)
        char_list = [_dict[text_id.item()] for text_id in text_index[batch_idx][selection]]
        if text_prob is not None:
            conf_list = text_prob[batch_idx][selection]
        else:
            conf_list = [1] * len(selection)
        if len(conf_list) == 0:
            conf_list = [0]

        text = ''.join(char_list)
        retval.append((text, np.mean(conf_list).tolist()))

    return retval


def _text_recognize(image: ImageTyping, model: str = 'ch_PP-OCRv4_rec',
                    is_remove_duplicate: bool = False) -> Tuple[str, float]:
    _ort_session = _open_ocr_recognition_model(model)
    expected_height = _ort_session.get_inputs()[0].shape[2]

    image = load_image(image, force_background='white', mode='RGB')
    r = expected_height / image.height
    new_height = int(round(image.height * r))
    new_width = int(round(image.width * r))
    image = image.resize((new_width, new_height))

    input_ = np.array(image).transpose((2, 0, 1)).astype(np.float32) / 255.0
    input_ = ((input_ - 0.5) / 0.5)[None, ...].astype(np.float32)
    _input_name = _ort_session.get_inputs()[0].name
    _output_name = _ort_session.get_outputs()[0].name
    output, = _ort_session.run([_output_name], {_input_name: input_})

    indices = output.argmax(axis=2)
    confs = output.max(axis=2)
    return _text_decode(indices, model, confs, is_remove_duplicate)[0]


@ts_lru_cache()
def _list_rec_models() -> List[str]:
    retval = []
    repo_segment_cnt = len(_REPOSITORY.split('/'))
    for item in _HF_CLIENT.glob(f'{_REPOSITORY}/rec/*/model.onnx'):
        retval.append(item.split('/')[repo_segment_cnt:][1])
    return retval

Code 'imgutils/ocr/__init__.py'

"""
Overview:
    Detect and recognize text in images.

    The models are exported from `PaddleOCR <https://github.com/PaddlePaddle/PaddleOCR>`_, hosted on
    `huggingface - deepghs/paddleocr <https://huggingface.co/deepghs/paddleocr/tree/main>`_.

    .. image:: ocr_demo.plot.py.svg
        :align: center

    This is an overall benchmark of all the text detection models:

    .. image:: ocr_det_benchmark.plot.py.svg
        :align: center

    and an overall benchmark of all the available text recognition models:

    .. image:: ocr_rec_benchmark.plot.py.svg
        :align: center

"""
from .entry import detect_text_with_ocr, ocr, list_det_models, list_rec_models

        ''')
    )
