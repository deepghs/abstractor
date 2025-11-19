import json
import os
from urllib.parse import urljoin

import numpy as np
import pandas as pd
from ditk import logging
from hfutils.operate import get_hf_client


def _safe(x):
    if isinstance(x, (list, tuple,)):
        return [_safe(y) for y in x]
    elif isinstance(x, np.ndarray):
        return [_safe(y) for y in x.tolist()]
    elif isinstance(x, dict):
        return {key: _safe(value) for key, value in x.items()}
    else:
        return x


def sync(repo_id: str, src_dir: str):
    hf_client = get_hf_client()
    df = pd.read_parquet(hf_client.hf_hub_download(
        repo_id=repo_id,
        repo_type='dataset',
        filename=f'members.parquet'
    ))

    with open(os.path.join(src_dir, 'app', 'json', 'members', 'core.json'), 'r') as f:
        core_usernames = [item['username'] for item in json.load(f)]
    with open(os.path.join(src_dir, 'app', 'json', 'members', 'partner.json'), 'r') as f:
        partner_usernames = [item['username'] for item in json.load(f)]

    df = df[~df['username'].isin(set(core_usernames) | set(partner_usernames))]
    df = df.sort_values(by=['num_papers', 'num_followers'], ascending=[False, False])
    logging.info(f'Members of DeepGHS:\n{df}')

    v = _safe(df.to_dict('records'))

    for item in v:
        item['name'] = item['fullname']
        item['type'] = 'normal'
        item['avatar'] = urljoin('https://huggingface.co', item['avatar'])

    with open(os.path.join(src_dir, 'app', 'json', 'members', 'normal.json'), 'w') as f:
        json.dump(v, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    logging.try_init_root(level=logging.INFO)
    sync(
        repo_id=os.environ['DS_REPO_ID'],
        src_dir='pages',
    )
