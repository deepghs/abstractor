from typing import Optional, Tuple

from hfutils.operate.base import RepoTypeTyping
from huggingface_hub import hf_hub_url
from huggingface_hub.utils import build_hf_headers

from .url import get_redirected_url


def hf_get_resource_url(repo_id: str, filename: str, repo_type: RepoTypeTyping = 'dataset', revision: str = 'main',
                        hf_token: Optional[str] = None) -> Tuple[str, Optional[int]]:
    return get_redirected_url(
        url=hf_hub_url(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=filename,
            revision=revision,
        ),
        headers=build_hf_headers(token=hf_token),
    )
