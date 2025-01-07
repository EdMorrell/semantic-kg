import string
import numpy as np
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
