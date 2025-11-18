import os
import time

import pandas as pd
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import get_hf_client, get_hf_fs, upload_directory_as_directory
from tqdm import tqdm

from abstractor.repository import ask_llm_for_hf_repo_info

BLACKLISTED_SPACES = [
    'deepghs/README',
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

    _last_update, has_update = None, False

    if hf_client.file_exists(
            repo_id=repository,
            repo_type='dataset',
            filename='spaces.parquet'
    ):
        df_spaces = pd.read_parquet(hf_client.hf_hub_download(
            repo_id=repository,
            repo_type='dataset',
            filename='spaces.parquet'
        ))
        origin_length = len(df_spaces)
        df_spaces = df_spaces[~df_spaces['repo_id'].isin(BLACKLISTED_SPACES)]
        # df_spaces = df_spaces[df_spaces['is_ready_to_view'].map(lambda x: x['yes'])]
        # df_spaces = df_spaces[df_spaces['is_clear_enough'].map(lambda x: x['yes'])]
        d_spaces = {item['repo_id']: item for item in df_spaces.to_dict('records')}
        if origin_length != len(d_spaces):
            has_update = True
    else:
        d_spaces = {}

    _last_space_count = len(d_spaces)

    def _deploy(force=False):
        nonlocal _last_update, has_update, _last_space_count

        if not has_update:
            return
        if not force and _last_update is not None and _last_update + deploy_span > time.time():
            return

        with TemporaryDirectory() as td:
            df_spaces = pd.DataFrame(list(d_spaces.values()))
            df_spaces = df_spaces.sort_values(by=['downloads'], ascending=[False])
            spaces_parquet_file = os.path.join(td, 'spaces.parquet')
            df_spaces.to_parquet(spaces_parquet_file, index=False)

            upload_directory_as_directory(
                repo_id=repository,
                repo_type='dataset',
                local_directory=td,
                path_in_repo='.',
                message=f'Add {plural_word(len(d_spaces) - _last_space_count, "space")}',
            )
            has_update = False
            _last_update = time.time()
            _last_space_count = len(d_spaces)

    for repo_item in tqdm(list(hf_client.list_spaces(author='deepghs'))):
        repo_id = repo_item.id
        if repo_id in BLACKLISTED_SPACES:
            continue
        if repo_item.private:
            continue
        if repo_id in d_spaces:
            continue

        logging.info(f'Repository: {repo_id!r}, repo_type: {"space"!r} ...')
        try:
            d_spaces[repo_id] = ask_llm_for_hf_repo_info(
                repo_id=repo_id,
                repo_type='space',
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
