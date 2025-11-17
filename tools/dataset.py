import os
import time

import pandas as pd
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import get_hf_client, get_hf_fs, upload_directory_as_directory
from tqdm import tqdm

from abstractor.repository import ask_llm_for_hf_repo_info

BLACKLISTED_DATASETS = [
    # 'deepghs/animefull-latest',
    # 'deepghs/animefull-latest-ckpt',
]


def sync(repository: str, deploy_span: float = 5 * 60):
    hf_client = get_hf_client()
    hf_fs = get_hf_fs()

    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True)
        attr_lines = hf_fs.read_text(f'datasets/{repository}/.gitattributes').splitlines(keepends=False)
        attr_lines.append('*.json filter=lfs diff=lfs merge=lfs -text')
        attr_lines.append('*.csv filter=lfs diff=lfs merge=lfs -text')
        hf_fs.write_text(
            f'datasets/{repository}/.gitattributes',
            os.linesep.join(attr_lines),
        )

    if hf_client.file_exists(
            repo_id=repository,
            repo_type='dataset',
            filename='datasets.parquet'
    ):
        df_datasets = pd.read_parquet(hf_client.hf_hub_download(
            repo_id=repository,
            repo_type='dataset',
            filename='datasets.parquet'
        ))
        df_datasets = df_datasets[~df_datasets['repo_id'].isin(BLACKLISTED_DATASETS)]
        # df_datasets = df_datasets[df_datasets['is_ready_to_view'].map(lambda x: x['yes'])]
        # df_datasets = df_datasets[df_datasets['is_clear_enough'].map(lambda x: x['yes'])]
        d_datasets = {item['repo_id']: item for item in df_datasets.to_dict('records')}
    else:
        d_datasets = {}

    _last_update, has_update = None, False
    _last_dataset_count = len(d_datasets)

    def _deploy(force=False):
        nonlocal _last_update, has_update, _last_dataset_count

        if not has_update:
            return
        if not force and _last_update is not None and _last_update + deploy_span > time.time():
            return

        with TemporaryDirectory() as td:
            df_datasets = pd.DataFrame(list(d_datasets.values()))
            df_datasets = df_datasets.sort_values(by=['downloads'], ascending=[False])
            datasets_parquet_file = os.path.join(td, 'datasets.parquet')
            df_datasets.to_parquet(datasets_parquet_file, index=False)

            upload_directory_as_directory(
                repo_id=repository,
                repo_type='dataset',
                local_directory=td,
                path_in_repo='.',
                message=f'Add {plural_word(len(d_datasets) - _last_dataset_count, "dataset")}',
            )
            has_update = False
            _last_update = time.time()
            _last_dataset_count = len(d_datasets)

    for repo_item in tqdm(list(hf_client.list_datasets(author='deepghs'))):
        repo_id = repo_item.id
        if repo_id in BLACKLISTED_DATASETS:
            continue
        if repo_item.private:
            continue
        if repo_id in d_datasets:
            continue

        logging.info(f'Repository: {repo_id!r}, repo_type: {"dataset"!r} ...')
        try:
            d_datasets[repo_id] = ask_llm_for_hf_repo_info(
                repo_id=repo_id,
                repo_type='dataset',
            )
            has_update = True
            _deploy(force=False)
        except:
            logging.exception(f'Skipped due to error - {repo_id!r}.')
            continue

    _deploy(force=True)


if __name__ == '__main__':
    logging.try_init_root(level=logging.INFO)
    sync(
        repository=os.environ['DS_REPO_ID']
    )
