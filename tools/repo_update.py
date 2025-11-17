import textwrap
from typing import Optional

from ditk import logging
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

**Model Repository Metadata Schema**:
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

**Dataset Repository Metadata Schema**:
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
````

#### 4. Critical Preservation Rules
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

#### 5. Special Case Handling
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

#### 6. Validation Checks
Before output, verify:
- Metadata block is valid YAML with required fields
- Summary contains ≥200 words and 3-5 bolded keywords
- All original tables/code blocks are preserved exactly
- License field exists in metadata (even if missing in body)
- No external commentary or markdown wrappers added)

**Output Format**: Raw README.md content only - no prefixes, suffixes, or explanations. Ready for direct use on Hugging Face Hub.
    """).strip()

    cnt = 0
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


if __name__ == '__main__':
    logging.try_init_root(level=logging.INFO)
    sync(
        repo_id='deepghs/nudenet_onnx',
        repo_type='model',
        extra_text="""

Code of dghs-imgutils, in 'imgutils/detect/nudenet.py':

\"\"\"
Overview:
    This module provides functionality for detecting nudity in images using the NudeNet model.
    
    The module includes functions for preprocessing images, running the NudeNet YOLO model,
    applying non-maximum suppression (NMS), and postprocessing the results. It utilizes
    ONNX models hosted on `deepghs/nudenet_onnx <https://huggingface.co/deepghs/nudenet_onnx>`_
    for efficient inference. The original project is
    `notAI-tech/NudeNet <https://github.com/notAI-tech/NudeNet>`_.
    
    .. collapse:: Overview of NudeNet Detect (NSFW Warning!!!)

        .. image:: nudenet_detect_demo.plot.py.svg
            :align: center
    
    The main function :func:`detect_with_nudenet` can be used to perform nudity detection on
    given images, returning a list of bounding boxes, labels, and confidence scores.
    
    This is an overall benchmark of all the nudenet models:

    .. image:: nudenet_detect_benchmark.plot.py.svg
        :align: center

    .. note::

        Here is a detailed list of labels from the NudeNet detection model and their respective meanings:

        .. list-table::
           :widths: 25 75
           :header-rows: 1

           * - Label
             - Description
           * - FEMALE_GENITALIA_COVERED
             - Detects covered female genitalia in the image.
           * - FACE_FEMALE
             - Detects the face of a female in the image.
           * - BUTTOCKS_EXPOSED
             - Detects exposed buttocks in the image.
           * - FEMALE_BREAST_EXPOSED
             - Detects exposed female breasts in the image.
           * - FEMALE_GENITALIA_EXPOSED
             - Detects exposed female genitalia in the image.
           * - MALE_BREAST_EXPOSED
             - Detects exposed male breasts in the image.
           * - ANUS_EXPOSED
             - Detects exposed anus in the image.
           * - FEET_EXPOSED
             - Detects exposed feet in the image.
           * - BELLY_COVERED
             - Detects a covered belly in the image.
           * - FEET_COVERED
             - Detects covered feet in the image.
           * - ARMPITS_COVERED
             - Detects covered armpits in the image.
           * - ARMPITS_EXPOSED
             - Detects exposed armpits in the image.
           * - FACE_MALE
             - Detects the face of a male in the image.
           * - BELLY_EXPOSED
             - Detects an exposed belly in the image.
           * - MALE_GENITALIA_EXPOSED
             - Detects exposed male genitalia in the image.
           * - ANUS_COVERED
             - Detects a covered anus in the image.
           * - FEMALE_BREAST_COVERED
             - Detects covered female breasts in the image.
           * - BUTTOCKS_COVERED
             - Detects covered buttocks in the image.


    .. note::
    
        This module requires onnxruntime version 1.18 or higher.
\"\"\"

from typing import Tuple, List

import numpy as np
from PIL import Image
from hbutils.testing.requires.version import VersionInfo
from huggingface_hub import hf_hub_download

from imgutils.data import ImageTyping
from imgutils.utils import open_onnx_model, ts_lru_cache
from ..data import load_image


def _check_compatibility() -> bool:
    \"\"\"
    Check if the installed onnxruntime version is compatible with NudeNet.

    :raises EnvironmentError: If the onnxruntime version is less than 1.18.
    \"\"\"
    import onnxruntime
    if VersionInfo(onnxruntime.__version__) < '1.18':
        raise EnvironmentError(f'Nudenet not supported on onnxruntime {onnxruntime.__version__}, '
                               f'please upgrade it to 1.18+ version.\n'
                               f'If you are running on CPU, use "pip install -U onnxruntime" .\n'
                               f'If you are running on GPU, use "pip install -U onnxruntime-gpu" .')  # pragma: no cover


_REPO_ID = 'deepghs/nudenet_onnx'


@ts_lru_cache()
def _open_nudenet_yolo():
    \"\"\"
    Open and cache the NudeNet YOLO ONNX model.

    :return: The loaded ONNX model for YOLO.
    \"\"\"
    return open_onnx_model(hf_hub_download(
        repo_id=_REPO_ID,
        repo_type='model',
        filename='320n.onnx',
    ))


@ts_lru_cache()
def _open_nudenet_nms():
    \"\"\"
    Open and cache the NudeNet NMS ONNX model.

    :return: The loaded ONNX model for NMS.
    \"\"\"
    return open_onnx_model(hf_hub_download(
        repo_id=_REPO_ID,
        repo_type='model',
        filename='nms-yolov8.onnx',
    ))


def _nn_preprocessing(image: ImageTyping, model_size: int = 320) -> Tuple[np.ndarray, float]:
    \"\"\"
    Preprocess the input image for the NudeNet model.

    :param image: The input image.
    :param model_size: The size to which the image should be resized (default: 320).
    :return: A tuple containing the preprocessed image array and the scaling ratio.
    \"\"\"
    image = load_image(image, mode='RGB', force_background='white')
    assert image.mode == 'RGB'
    mat = np.array(image)

    max_size = max(image.width, image.height)

    mat_pad = np.zeros((max_size, max_size, 3), dtype=np.uint8)
    mat_pad[:mat.shape[0], :mat.shape[1], :] = mat
    img_resized = Image.fromarray(mat_pad, mode='RGB').resize((model_size, model_size), resample=Image.BILINEAR)

    input_data = np.array(img_resized).transpose(2, 0, 1).astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    return input_data, max_size / model_size


def _make_np_config(topk: int = 100, iou_threshold: float = 0.45, score_threshold: float = 0.25) -> np.ndarray:
    \"\"\"
    Create a configuration array for the NMS model.

    :param topk: The maximum number of detections to keep (default: 100).
    :param iou_threshold: The IoU threshold for NMS (default: 0.45).
    :param score_threshold: The score threshold for detections (default: 0.25).
    :return: A numpy array containing the configuration parameters.
    \"\"\"
    return np.array([topk, iou_threshold, score_threshold]).astype(np.float32)


def _nn_postprocess(selected, global_ratio: float):
    \"\"\"
    Postprocess the model output to generate bounding boxes and labels.

    :param selected: The output from the NMS model.
    :param global_ratio: The scaling ratio to apply to the bounding boxes.
    :return: A list of tuples, each containing a bounding box, label, and confidence score.
    \"\"\"
    bboxes = []
    num_boxes = selected.shape[0]
    for idx in range(num_boxes):
        data = selected[idx, :]

        scores = data[4:]
        score = np.max(scores)
        label = np.argmax(scores)

        box = data[:4] * global_ratio
        x = (box[0] - 0.5 * box[2]).item()
        y = (box[1] - 0.5 * box[3]).item()
        w = box[2].item()
        h = box[3].item()

        bboxes.append(((x, y, x + w, y + h), _LABELS[label], score.item()))

    return bboxes


_LABELS = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED"
]


def detect_with_nudenet(image: ImageTyping, topk: int = 100,
                        iou_threshold: float = 0.45, score_threshold: float = 0.25) \
        -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    \"\"\"
    Detect nudity in the given image using the NudeNet model.

    :param image: The input image to analyze.
    :param topk: The maximum number of detections to keep (default: 100).
    :param iou_threshold: The IoU threshold for NMS (default: 0.45).
    :param score_threshold: The score threshold for detections (default: 0.25).
    :return: A list of tuples, each containing:

             - A bounding box as (x1, y1, x2, y2)
             - A label string
             - A confidence score
    \"\"\"
    _check_compatibility()
    input_, global_ratio = _nn_preprocessing(image, model_size=320)
    config = _make_np_config(topk, iou_threshold, score_threshold)
    output0, = _open_nudenet_yolo().run(['output0'], {'images': input_})
    selected, = _open_nudenet_nms().run(['selected'], {'detection': output0, 'config': config})
    return _nn_postprocess(selected[0], global_ratio=global_ratio)


        """
    )
