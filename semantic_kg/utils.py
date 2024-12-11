import numpy as np
import networkx as nx


def softmax(input_arr: np.ndarray, temperature: float) -> np.ndarray:
    return (
        np.exp(input_arr / temperature)
        / np.exp(input_arr / temperature).sum()
    )


def graph_edit_distance(g1: nx.Graph, g2: nx.Graph) -> int:
    def _edge_equivalence(edge1: dict, edge2: dict) -> bool:
        return edge1 == edge2

    return nx.graph_edit_distance(
        g1, g2, edge_match=_edge_equivalence
    )
