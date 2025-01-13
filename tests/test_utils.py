import networkx as nx

from semantic_kg import utils


def test_compute_edit_distance_edge_replacement() -> None:
    g1 = nx.Graph()
    g1.add_node("A", id="A")
    g1.add_node("B", id="B")
    g1.add_edge("A", "B", name="increase")

    g2 = nx.Graph()
    g2.add_node("A", id="A")
    g2.add_node("B", id="B")
    g2.add_edge("A", "B", name="decrease")

    assert utils.compute_edit_distance(g1, g2, "id", "name") == 1


def test_compute_edit_distance_node_replacement() -> None:
    g1 = nx.Graph()
    g1.add_node("A", id="A")
    g1.add_node("B", id="B")
    g1.add_edge("A", "B", name="increase")

    g2 = nx.Graph()
    g2.add_node("A", id="A")
    g2.add_node("C", id="C")
    g2.add_edge("A", "C", name="increase")

    assert utils.compute_edit_distance(g1, g2, "id", "name") == 1


def test_compute_edit_distance_multi_edit() -> None:
    g1 = nx.Graph()
    g1.add_node("A", id="A")
    g1.add_node("B", id="B")
    g1.add_node("C", id="C")
    g1.add_edge("A", "B", name="increase")
    g1.add_edge("B", "C", name="increase")

    g2 = nx.Graph()
    g2.add_node("A", id="A")
    g2.add_node("B", id="B")
    g2.add_node("C", id="C")
    # We replace one edge and remove another
    g2.add_edge("A", "B", name="decrease")

    assert utils.compute_edit_distance(g1, g2, "id", "name") == 2
