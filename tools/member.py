import os
import time

import pandas as pd
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import get_hf_client, get_hf_fs, upload_directory_as_directory

from abstractor.repository import create_user_info


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
            filename='members.parquet'
    ):
        df_members = pd.read_parquet(hf_client.hf_hub_download(
            repo_id=repository,
            repo_type='dataset',
            filename='members.parquet'
        ))
        d_members = {item['username']: item for item in df_members.to_dict('records')}
    else:
        d_members = {}

    _last_update, has_update = None, False
    _last_member_count = len(d_members)

    def _deploy(force=False):
        nonlocal _last_update, has_update, _last_member_count

        if not has_update:
            return
        if not force and _last_update is not None and _last_update + deploy_span > time.time():
            return

        with TemporaryDirectory() as td:
            df_members = pd.DataFrame(list(d_members.values()))
            df_members = df_members.sort_values(by=['num_followers', 'username'], ascending=[False, True])
            members_parquet_file = os.path.join(td, 'members.parquet')
            df_members.to_parquet(members_parquet_file, index=False)

            upload_directory_as_directory(
                repo_id=repository,
                repo_type='dataset',
                local_directory=td,
                path_in_repo='.',
                message=f'Add {plural_word(len(d_members) - _last_member_count, "member")}',
            )
            has_update = False
            _last_update = time.time()
            _last_member_count = len(d_members)

    for member_item in hf_client.list_organization_members(organization='deepghs'):
        username = member_item.username
        if username in d_members:
            continue

        logging.info(f'Username: {username!r} ...')
        try:
            d_members[username] = create_user_info(
                author=username,
            )
            has_update = True
            _deploy(force=False)
        except:
            logging.exception(f'Skipped due to error - {username!r}.')
            continue

    _deploy(force=True)


if __name__ == '__main__':
    logging.try_init_root(level=logging.INFO)
    sync(
        repository=os.environ['DS_REPO_ID']
    )
