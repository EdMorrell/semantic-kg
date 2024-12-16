from pathlib import Path
from typing import Optional

import hishel


def get_hishel_http_client(cache_dir: Optional[str | Path] = None):
    """Wraps an HTTP client to enable caching"""
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)

    controller = hishel.Controller(force_cache=True, cacheable_methods=["GET", "POST"])
    storage = hishel.FileStorage(base_path=cache_dir)
    http_client = hishel.CacheClient(controller=controller, storage=storage)

    return http_client
