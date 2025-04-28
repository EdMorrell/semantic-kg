import re
import ast
import string
from typing import Optional
from pathlib import Path

import hishel
import numpy as np
import pandas as pd
import networkx as nx


def softmax(input_arr: np.ndarray, temperature: float) -> np.ndarray:
    return np.exp(input_arr / temperature) / np.exp(input_arr / temperature).sum()


def compute_edit_distance(
    graph: nx.Graph, p_graph: nx.Graph, node_id_field: str, edge_id_field: str
) -> float:
    def _node_match(n1: dict, n2: dict) -> bool:
        return n1[node_id_field] == n2[node_id_field]

    def _edge_match(e1: dict, e2: dict) -> bool:
        return e1[edge_id_field] == e2[edge_id_field]

    return nx.graph_edit_distance(
        graph,
        p_graph,
        node_match=_node_match,
        edge_match=_edge_match,
    )


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


def load_subgraph_dataset(fpath: Path | str) -> pd.DataFrame:
    subgraph_dataset = pd.read_csv(str(fpath))

    transform_cols = [
        "subgraph_triples",
        "perturbed_subgraph_triples",
        "perturbation_log",
    ]
    for col in transform_cols:
        # Removes any np.str_ formatted strings
        subgraph_dataset[col] = subgraph_dataset[col].apply(
            lambda x: re.sub(r"np\.str_\(\'(.+?)\'\)", r"'\1'", x)
        )
        subgraph_dataset[col] = subgraph_dataset[col].apply(ast.literal_eval)

    return subgraph_dataset
