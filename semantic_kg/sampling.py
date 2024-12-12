import random
from typing import Any, Literal, Optional
from collections import deque

import numpy as np
import networkx as nx


def bfs(
    graph: nx.Graph,
    start_node: str | int,
    n_nodes: int,
    max_neighbours: Optional[int] = None,
) -> list[tuple[str, str] | tuple[int, int]]:
    """Performs BFS on a graph returning node-pairs

    Breadth-first search on a graph. Optionally only explore
    a subset of a node's neighbors to encourage a mixture of
    breadth and depth

    Parameters
    ----------
    graph : nx.Graph
        Graph to sample from
    start_node : str | int
        First node to sample from
    n_nodes : int
        Number of nodes to sample in final sub-graph
    max_neighbours : int
        Optionally explore a maximum number of neighbors for each node.
        If None then method is equivalent to BFS.

    Returns
    -------
    list[tuple[str, str] | tuple[int, int]]
        List of node-pairs explored via search method
    """
    visited = set()
    queue = deque()
    output = []

    visited.add(start_node)
    queue.append(start_node)

    while queue and len(visited) < n_nodes:
        current = queue.popleft()

        edges = list(graph.edges(current))
        if max_neighbours:
            edges = edges[:max_neighbours]

        for edge in edges:
            ref = edge[1]
            if ref not in visited:
                visited.add(ref)
                queue.append(ref)
                output.append((current, ref))

    return output


def bfs_node_diversity(
    graph: nx.Graph,
    start_node: str | int,
    n_nodes: int,
    max_neighbours: int,
    node_type_field: str,
) -> list[tuple[str, str] | tuple[int, int]]:
    """Variant of BFS that encourages diversity in nodes of final output

    Method decays the selection probability of previously selected node-types
    encouraging more node-type diversity in final sub-graph

    Parameters
    ----------
    graph : nx.Graph
        Graph to sample from
    start_node : str | int
        First node to sample from
    n_nodes : int
        Number of nodes to sample in final sub-graph
    max_neighbours : int
        Maximum number of neighbors to explore for each node
    node_type_field : str
        Field that indicates "node-type". Used to ensure selection probability
        of a node with the same "node-type" is reduced with each iteration

    Returns
    -------
    list[tuple[str, str] | tuple[int, int]]
        List of node-pairs explored via search method
    """
    visited = set()
    queue = deque()
    output = []
    decay_factor = 1 - (1 / n_nodes)

    unique_node_types = list(
        set([n[1] for n in list(graph.nodes.data(node_type_field))])  # type: ignore
    )
    node_pvals = np.ones(len(unique_node_types)) / len(unique_node_types)
    pval_idx_map = dict(zip(unique_node_types, np.arange(len(unique_node_types))))

    visited.add(start_node)
    queue.append(start_node)
    while queue and len(visited) < n_nodes:
        current = queue.popleft()

        neighbors = list(graph.neighbors(current))
        neighbor_pvals = np.array(
            [
                node_pvals[pval_idx_map[graph.nodes[n][node_type_field]]]
                for n in neighbors
            ]
        )
        # Renormalize probs
        neighbor_pvals /= neighbor_pvals.sum()

        n_samples = min(len(neighbors), max_neighbours)
        edges = np.random.choice(
            neighbors, size=n_samples, replace=False, p=neighbor_pvals
        )

        for edge in edges:
            if edge not in visited:
                visited.add(edge)
                queue.append(edge)
                output.append((current, edge))

                # Adjust selection probability
                p_idx = pval_idx_map[graph.nodes[edge][node_type_field]]
                node_pvals[p_idx] *= decay_factor
                node_pvals /= node_pvals.sum()

    return output


class SubgraphSampler:
    METHOD_MAP = {"bfs": bfs, "bfs_node_diversity": bfs_node_diversity}

    def __init__(
        self,
        graph: nx.Graph,
        node_name_field: str,
        edge_name_field: str,
        method: Literal["bfs", "bfs_node_diversity"] = "bfs",
    ) -> None:
        self.g = graph
        self.method = method
        self.node_name_field = node_name_field
        self.edge_name_field = edge_name_field
        if method not in self.METHOD_MAP:
            raise ValueError(f"Invalid method: {method}")

    def sample(
        self,
        n_nodes: int,
        start_node: Optional[str | int] = None,
        max_neighbours: int = 5,
        **kwargs,
    ) -> nx.Graph:
        if not start_node:
            start_node = random.choice(self.g.nodes)["node_index"]

        subgraph_edges = SubgraphSampler.METHOD_MAP[self.method](
            self.g, start_node, n_nodes, max_neighbours, **kwargs
        )
        nodes = [item for t in subgraph_edges for item in t]
        return self.g.subgraph(nodes)


def generate_triplets(
    graph: nx.Graph,
    node_name_field: str,
    edge_name_field: str,
    node_attr_fields: Optional[list[str]] = None,
    edge_attr_fields: Optional[list[str]] = None,
) -> list[dict[str, str]]:
    def _add_meta_to_dict(
        triplet_dict: dict[str, str], attr_fields: list[str], metadata: dict[str, Any]
    ) -> dict[str, str | dict]:
        triplet_dict["meta"] = {}  # type: ignore
        for field in attr_fields:
            triplet_dict["meta"][field] = metadata[field]

        return triplet_dict  # type: ignore

    triplets = []
    for edge in graph.edges:
        source_node = {"name": graph.nodes[edge[0]][node_name_field]}
        if node_attr_fields:
            source_node = _add_meta_to_dict(
                source_node, node_attr_fields, graph.nodes[edge[0]]
            )

        edge_dict = {"name": graph[edge[0]][edge[1]][edge_name_field]}
        if edge_attr_fields:
            edge_dict = _add_meta_to_dict(
                edge_dict, edge_attr_fields, graph[edge[0]][edge[1]]
            )

        target_node = {"name": graph.nodes[edge[1]][node_name_field]}
        if node_attr_fields:
            target_node = _add_meta_to_dict(
                target_node, node_attr_fields, graph.nodes[edge[1]]
            )

        triplet = {
            "source_node": source_node,
            "relation": edge_dict,
            "target_node": target_node,
        }
        triplets.append(triplet)

    return triplets
