import logging
from typing import Optional

import duckdb
from hfutils.operate.base import RepoTypeTyping
from hfutils.utils import hf_fs_path

from ..utils import hf_get_resource_url


def sample_from_parquet(repo_id: str, filename: str, repo_type: RepoTypeTyping = 'dataset', revision: str = 'main',
                        hf_token: Optional[str] = None, max_rows: int = 5):
    fs_path = hf_fs_path(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=filename,
        revision=revision,
    )
    logging.info(f'Sampling parquet file {fs_path!r} ...')
    url, content_size = hf_get_resource_url(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=filename,
        revision=revision,
        hf_token=hf_token,
    )
    conn = duckdb.connect(":memory:")
    try:
        conn.execute("INSTALL httpfs")
        conn.execute("LOAD httpfs")

        # Read Parquet with DuckDB (uses HTTP range requests automatically!)
        query = f"SELECT * FROM read_parquet('{url}') LIMIT {max_rows}"

        result = conn.execute(query).fetchall()
        columns = [desc[0] for desc in conn.description]

        # Get total row count (efficient - reads only metadata)
        try:
            count_query = f"SELECT COUNT(*) FROM read_parquet('{url}')"
            total_rows = conn.execute(count_query).fetchone()[0]
        except Exception:
            logging.exception('Error when counting this table ...')
            total_rows = len(result)

    finally:
        conn.close()

    return {
        "columns": columns,
        "rows": [dict(zip(columns, list(row))) for row in result],
        "total_rows": total_rows,
        "truncated": len(result) >= max_rows or total_rows > max_rows,
        "file_size": content_size,
    }
