import io
import logging
import textwrap
from pprint import pformat
from typing import List, Optional

from hfutils.operate import get_hf_client
from tqdm import tqdm

from abstractor.openai import ask_llm
from abstractor.utils import parse_json_from_llm_output
from .tree import ask_llm_for_hf_repo_info


def pick_user_repos(author: str, org_names: Optional[List[str]] = None, max_retries: int = 5,
                    max_selected: int = 10, hf_token: Optional[str] = None):
    org_names = list(org_names or [])
    hf_client = get_hf_client(hf_token=hf_token)

    models, datasets, spaces = [], [], []
    for ax in [author, *org_names]:
        for repo_item in hf_client.list_models(author=ax):
            if not repo_item.private:
                models.append(repo_item.id)
        for repo_item in hf_client.list_datasets(author=ax):
            if not repo_item.private:
                datasets.append(repo_item.id)
        for repo_item in hf_client.list_spaces(author=ax):
            if not repo_item.private:
                spaces.append(repo_item.id)

    logging.info(f'Author: {author!r}\n'
                 f'Models: {models!r}\n'
                 f'Datasets: {datasets!r}\n'
                 f'Spaces: {spaces!r}')

    _SYSTEM_PROMPT = textwrap.dedent(f"""
You are an AI assistant specialized in analyzing Hugging Face user profiles and repositories. Your task is to select the most noteworthy repositories from a user's profile across three categories (models, datasets, spaces) based strictly on the provided user information and repository lists. Follow these rules:

1. **Input Analysis**:
   - You will receive:
     a) User profile data (username, metadata, API details)
     b) Three lists of repository IDs: `models`, `datasets`, and `spaces` (exactly as provided)
   - Only consider repositories explicitly provided in the input lists

2. **Selection Criteria**:
   For each category (models/datasets/spaces):
   - Prioritize repositories that best represent:
     • Technical expertise & research direction
     • Community impact (likes/downloads/stars)
     • Recent activity (updated within last 12 months)
     • Uniqueness/innovation
   - If a category has ≤{max_selected} repos: select all
   - If a category has >{max_selected} repos: select {max_selected} most significant based on above criteria
   - Never select repos outside provided lists

3. **Output Requirements**:
   - Output ONLY pure JSON format with EXACT structure:
     {{
       "datasets": ["repo_id1", "repo_id2", ...],  // Max {max_selected}
       "models": ["repo_id3", "repo_id4", ...],    // Max {max_selected}
       "spaces": ["repo_id5", ...]                 // Max {max_selected}
     }}
   - Each string must be a verbatim repo_id from input
   - No additional text, explanations, or formatting
   - Empty lists if no repos exist in a category
   - Maintain original repo_id casing

Failure to comply will cause system errors. Confirm output contains ONLY valid JSON parsable by `json.loads()`.
    """).strip()

    with io.StringIO() as sf:
        print(f'Huggingface User: {author}', file=sf)
        print(f'Noticed Orgs: {org_names!r}', file=sf)
        user_info = hf_client.get_user_overview(username=author)
        if user_info.details:
            print(f'User details: {user_info.details!r}', file=sf)
        print(f'Profile from API: {user_info!r}', file=sf)
        print(f'', file=sf)
        print(f'Noticed models: {models!r}', file=sf)
        print(f'Noticed datasets: {datasets!r}', file=sf)
        print(f'Noticed spaces: {spaces!r}', file=sf)
        print(f'', file=sf)

        cnt = 0
        while True:
            try:
                selected_repos = parse_json_from_llm_output(ask_llm([
                    {
                        "role": "system",
                        "content": _SYSTEM_PROMPT,
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

        logging.info(f'Expected filename: {selected_repos!r}')
        return selected_repos


def create_user_info(author: str, org_names: Optional[List[str]] = None, max_retries: int = 5,
                     max_selected: int = 10, hf_token: Optional[str] = None):
    org_names = list(org_names or [])
    hf_client = get_hf_client(hf_token=hf_token)

    models, datasets, spaces = [], [], []
    for ax in [author, *org_names]:
        for repo_item in hf_client.list_models(author=ax):
            if not repo_item.private:
                models.append(repo_item.id)
        for repo_item in hf_client.list_datasets(author=ax):
            if not repo_item.private:
                datasets.append(repo_item.id)
        for repo_item in hf_client.list_spaces(author=ax):
            if not repo_item.private:
                spaces.append(repo_item.id)

    logging.info(f'Author: {author!r}\n'
                 f'Models: {models!r}\n'
                 f'Datasets: {datasets!r}\n'
                 f'Spaces: {spaces!r}')

    _SYSTEM_PROMPT = textwrap.dedent("""
SYSTEM PROMPT:
**Objective**: Generate a structured JSON profile for a Hugging Face user using ONLY provided raw data. Output MUST be pure JSON without any prefixes/suffixes/explanations.

### Input Data Structure:
1. `user_info`: Dict containing user metadata from HuggingFace API (username, fullName, location, avatarUrl, bioText, socialLinks, etc.)
2. `repositories`: List of user's models/datasets/spaces with extracted summaries

### Output Requirements:
{
  "name": "username from user_info",
  "role": "[Developer|Data Engineer|AI Researcher|Other]",
  "type": "normal",
  "avatar": "direct avatarUrl from user_info",
  "location": "null or inferred location ONLY if explicitly supported",
  "detail": {
    "social": [{platform, url}],
    "expertise": ["Capitalized", "Relevant", "Terms"],
    "sign": "EXACT user quote or null",
    "bio": "40-100 word professional summary"
  }
}

### Critical Instructions:
1. **Bio Composition**:
   - Length: Strictly 40-100 words (if there is really something to say)
   - Content: Integrate ALL elements below based on repository summaries and user_info:
     * Professional introduction
     * Work focus areas
     * Research directions
     * Technical methodologies
     * Open-source contributions
   - Source: Derive ONLY from repository summaries and user_info
   - Style: Professional/technical tone
   - If there is really nothing to talk about of this user (e.g. some empty user), 
     just leave only one sentence like 'XXX is a contributor on Huggingface, member of DeepGHS'.
     You do not need to follow the 40-100 words limitation in these cases.

2. **Sign Field**:
   - Extract VERBATIM from `user_info.bioText` if present
   - If no signature-like quote exists: "sign": null
   - NEVER invent/create/imply signatures

3. **Expertise**:
   - Generate 3-5 capitalized terms (e.g., "Transformer Architectures")
   - MUST align with repository topics and bio content
   - Validated against: Model types, dataset subjects, and space functionalities
   - If there is really nothing to talk about of this user (e.g. some empty user), just leave an empty list.

4. **Avatar & Name**:
   - avatar: Directly use `user_info.avatarUrl`
   - name: `user_info.username` ONLY (ignore fullName)

5. **Location Handling**:
   - Set to "null" unless ALL conditions met:
     a) Explicitly stated in `user_info.location`
     b) Contains country/region name
     c) Supported by repository metadata (e.g., geo-specific datasets)
   - NEVER infer from names/linguistic cues

6. **Role Assignment**:
   - Analyze repository types:
     • ≥70% models: "AI Researcher"
     • ≥70% datasets: "Data Engineer"
     • Spaces/code-heavy repos: "Developer"
     • Mixed repos: "Developer" unless research-focused
     • If there is really nothing to talk about of this user (e.g. some empty user): "Contributor"
   - Default: "Developer"

7. **Social Links**:
   - Prioritized platforms in order:
     1. HuggingFace (MUST include: `https://huggingface.co/<username>`)
     2. GitHub (if in user_info.socialLinks)
     3. LinkedIn/Discord/X/Bluesky/Civitai
     4. Personal/GitHub Pages
   - Format: {platform: "PlatformName", url: "raw_url"}
   - Include ONLY links present in input data

8. **Prohibitions**:
   - NO markdown/output formatting
   - NO assumptions beyond provided data
   - NO translation/interpretation of user content
   - NO location guessing (null if ambiguous)
   - NO role assignment beyond specified cases

### Processing Workflow:
1. Extract RAW values from user_info first
2. Analyze repositories to determine:
   - Technical focus (expertise)
   - Role category
   - Project themes (for bio)
3. Compose bio using repository summaries ONLY
4. Validate expertise against repository topics
5. Filter social links using platform priority list
6. Enforce word count for bio (40-100 words)

### Output Enforcement:
- If ANY required field is missing from input: Use "null"
- FINAL OUTPUT MUST BE VALID JSON PARSEABLE BY `json.loads()`

    """).strip()

    with io.StringIO() as sf:
        print(f'Huggingface User: {author}', file=sf)
        user_api_info = hf_client.get_user_overview(username=author)
        print(f'Username: {user_api_info.username!r}', file=sf)
        print(f'Full Name: {user_api_info.fullname!r}', file=sf)
        print(f'Noticed Orgs: {org_names!r}', file=sf)

        if user_api_info.details:
            print(f'User details: {user_api_info.details!r}', file=sf)
        print(f'Profile from API: {user_api_info!r}', file=sf)
        print(f'', file=sf)
        print(f'Noticed models: {models!r}', file=sf)
        print(f'Noticed datasets: {datasets!r}', file=sf)
        print(f'Noticed spaces: {spaces!r}', file=sf)
        print(f'', file=sf)

        picked = pick_user_repos(author, org_names, max_retries, max_selected, hf_token)
        if picked.get('models'):
            print(f'# Models', file=sf)
            print(f'', file=sf)
            for repo_id in tqdm(picked['models'], desc=f'Models of {author}'):
                if hf_client.repo_exists(
                        repo_id=repo_id,
                        repo_type='model',
                ):
                    logging.info(f'Scanning repository {repo_id!r}, repo_type: model ...')
                    print(f'## {repo_id}', file=sf)
                    print(f'', file=sf)
                    print(pformat(ask_llm_for_hf_repo_info(
                        repo_id=repo_id,
                        repo_type='model',
                        hf_token=hf_token,
                    )), file=sf)
                    print(f'', file=sf)

        if picked.get('datasets'):
            print(f'# Datasets', file=sf)
            print(f'', file=sf)
            for repo_id in tqdm(picked['datasets'], desc=f'Dataset of {author}'):
                if hf_client.repo_exists(
                        repo_id=repo_id,
                        repo_type='dataset',
                ):
                    logging.info(f'Scanning repository {repo_id!r}, repo_type: dataset ...')
                    print(f'## {repo_id}', file=sf)
                    print(f'', file=sf)
                    print(pformat(ask_llm_for_hf_repo_info(
                        repo_id=repo_id,
                        repo_type='dataset',
                        hf_token=hf_token,
                    )), file=sf)
                    print(f'', file=sf)

        if picked.get('spaces'):
            print(f'# Spaces', file=sf)
            print(f'', file=sf)
            for repo_id in tqdm(picked['spaces'], desc=f'Spaces of {author}'):
                if hf_client.repo_exists(
                        repo_id=repo_id,
                        repo_type='space',
                ):
                    logging.info(f'Scanning repository {repo_id!r}, repo_type: space ...')
                    print(f'## {repo_id}', file=sf)
                    print(f'', file=sf)
                    print(pformat(ask_llm_for_hf_repo_info(
                        repo_id=repo_id,
                        repo_type='space',
                        hf_token=hf_token,
                    )), file=sf)
                    print(f'', file=sf)

        cnt = 0
        while True:
            try:
                user_info = parse_json_from_llm_output(ask_llm([
                    {
                        "role": "system",
                        "content": _SYSTEM_PROMPT,
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

        user_info = {
            **user_info,
            'username': user_api_info.username,
            'fullname': user_api_info.fullname,
            'num_following': user_api_info.num_following,
            'num_followers': user_api_info.num_followers,
            'num_datasets': user_api_info.num_datasets,
            'num_models': user_api_info.num_models,
            'num_spaces': user_api_info.num_spaces,
            'num_papers': user_api_info.num_papers,
        }
        logging.info(f'Expected filename: {user_info!r}')
        return user_info
