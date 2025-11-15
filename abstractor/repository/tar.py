import logging
import tarfile
from typing import Optional

from fsspec.implementations.http import HTTPFileSystem
from hfutils.operate.base import RepoTypeTyping
from hfutils.utils import hf_fs_path

from ..utils import hf_get_resource_url


def sample_from_tar(repo_id: str, filename: str, repo_type: RepoTypeTyping = 'dataset', revision: str = 'main',
                    hf_token: Optional[str] = None, max_files: int = 100):
    fs_path = hf_fs_path(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=filename,
        revision=revision,
    )
    logging.info(f'Sampling parquet file {fs_path!r} ...')
    url = hf_get_resource_url(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=filename,
        revision=revision,
        hf_token=hf_token,
    )

    fs = HTTPFileSystem()

    # Open TAR file with fsspec (streaming, no full download)
    with fs.open(url, mode="rb") as f:
        files = []
        total_size = 0

        with tarfile.open(fileobj=f, mode="r|*") as tar:
            for member in tar:
                if member.isfile():
                    files.append(
                        {
                            "name": member.name,
                            "size": member.size,
                            "type": "file",
                        }
                    )
                    total_size += member.size

                    # Limit number of files listed
                    if len(files) >= max_files:
                        break

        return {
            "files": files,
            "total_files": len(files),
            "total_size": total_size,
            "truncated": len(files) >= max_files,
        }
