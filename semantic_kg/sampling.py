import random
from typing import Any, Literal, Optional
from collections import deque

import networkx as nx


def bfs(
    graph: nx.Graph,
    start_node: str | int,
    n_nodes: int,
    max_neighbours: Optional[int] = None,
) -> list[tuple[str, str] | tuple[int, int]]:
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


class SubgraphSampler:
    METHOD_MAP = {"bfs": bfs}

    def __init__(
        self,
        graph: nx.Graph,
        node_name_field: str,
        edge_name_field: str,
        method: Literal["bfs"] = "bfs",
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
    ) -> nx.Graph:
        if not start_node:
            start_node = random.choice(self.g.nodes)["node_index"]

        subgraph_edges = SubgraphSampler.METHOD_MAP[self.method](
            self.g, start_node, n_nodes, max_neighbours  # type: ignore
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
