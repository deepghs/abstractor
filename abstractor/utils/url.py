from typing import Optional, Dict, Tuple

import requests


# https://github.com/KohakuBlueleaf/KohakuHub/blob/c3e4201b77b4c93f72b3d9cb7dd0dad8e1318207/src/kohakuhub/datasetviewer/parsers.py#L93
def get_redirected_url(url, headers: Optional[Dict[str, str]] = None) -> Tuple[str, Optional[int]]:
    headers = dict(headers or {})
    resp = requests.head(url, headers=headers, allow_redirects=True)
    resp.raise_for_status()
    content_length = resp.headers.get('Content-Length')
    if content_length:
        content_length = int(content_length)
    return resp.url, content_length
