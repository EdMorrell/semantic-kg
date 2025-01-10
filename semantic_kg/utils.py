from typing import Optional
from pathlib import Path
import string

import hishel
import numpy as np
import networkx as nx


def softmax(input_arr: np.ndarray, temperature: float) -> np.ndarray:
    return np.exp(input_arr / temperature) / np.exp(input_arr / temperature).sum()


def graph_edit_distance(g1: nx.Graph, g2: nx.Graph) -> int:
    def _edge_equivalence(edge1: dict, edge2: dict) -> bool:
        return edge1 == edge2

    return nx.graph_edit_distance(g1, g2, edge_match=_edge_equivalence)


def find_field_placeholders(text: str) -> list[str]:
    """Finds field placeholders in unformatted strings.

    NOTE: double brackets don't work.
    """
    placeholders = [item[1] for item in string.Formatter().parse(text) if item[1]]
    return placeholders


def get_hishel_http_client(cache_dir: Optional[str | Path] = None):
    """Wraps an HTTP client to enable caching"""
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)

    controller = hishel.Controller(force_cache=True, cacheable_methods=["GET", "POST"])
    storage = hishel.FileStorage(base_path=cache_dir)
    http_client = hishel.CacheClient(controller=controller, storage=storage)

    return http_client
