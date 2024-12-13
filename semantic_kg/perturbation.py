import abc
import random
from typing import Optional, Protocol

import numpy as np
import networkx as nx

from semantic_kg import utils


class NoValidEdgeError(Exception):
    """Error for instance where no valid edges available for a perturbation"""

    pass


class BasePerturbation(abc.ABC):
    def __init__(self) -> None:
        """Base class for a perturbation operation to apply to a graph"""
        self.perturbation_log = []

        # Counts the edits made by a perturbation. Any edit is a single
        # change made to an edge or node.
        # NOTE: Node perturbations typically constitute >1 edit due to
        # their impact on surrounding edges
        self.edit_count = 0

    def _log_perturbation(
        self,
        perturbation_type: str,
        src_node: int | str,
        target_node: Optional[int | str],
        metadata: Optional[dict] = None,
    ) -> None:
        """Logs the type of pertubation made"""
        perturbation = {
            "type": perturbation_type,
            "source": src_node,
            "target": target_node,
            "metadata": metadata,
        }
        self.perturbation_log.append(perturbation)

    @abc.abstractmethod
    def create(self, graph: nx.Graph) -> nx.Graph:
        """Method should implement the perturbation"""
        pass


class EdgeAttributeMapper(Protocol):
    @abc.abstractmethod
    def get_attributes(
        self, src_node: dict, target_node: dict, edge_value: Optional[str] = None
    ) -> dict:
        pass


class EdgeAdditionPerturbation(BasePerturbation):
    def __init__(
        self,
        node_name_id: str,
        valid_edges: Optional[list[str]] = None,
        edge_attribute_mapper: Optional[EdgeAttributeMapper] = None,
        temperature: float = 0.2,
    ) -> None:
        """Applies a perturbation to the graph by adding an edge

        Parameters:
        ----------
        node_name_id : str
            The ID field of the node in the graph
        valid_edges : Optional[list[str]], optional
            A list of valid edges, if none set then all edges valid, by
            default None.
        edge_attribute_mapper : Optional[EdgeAttributeMapper], optional
            An optional object to use for generating edge attributes,
            by default None.
        temperature : float, optional
            The create method uses a softmax function to sample the
            nodes to connect to based on the distance to that edge.
            The temperature parameter controls the randomness of the
            sampling, by default 0.2, which biases sampling towards
            closer nodes. To allow edges between more distant nodes,
            increase the temperature.
        """
        super().__init__()
        self.node_name_id = node_name_id
        self.valid_edges = valid_edges
        self.edge_attribute_mapper = edge_attribute_mapper
        self.temperature = temperature

    def create(self, graph: nx.Graph, max_resamples: int = 10) -> nx.Graph:
        src_node = random.choice(list(graph.nodes))

        all_distance = nx.floyd_warshall(graph)
        node_distances = {k: v for k, v in all_distance[src_node].items() if v > 1}
        distances = np.array(list(node_distances.values()))

        inv_distances = 1 / distances

        pvals = utils.softmax(inv_distances, self.temperature)

        resamples = 0
        while resamples < max_resamples:
            target_node = list(node_distances.keys())[
                np.where(np.random.multinomial(1, pvals))[0][0]
            ]

            src_node_name = graph.nodes[src_node][self.node_name_id]
            target_node_name = graph.nodes[target_node][self.node_name_id]
            edge_name = f"{src_node_name}_{target_node_name}"

            # Ensures only a valid pair is found
            if not self.valid_edges or edge_name in self.valid_edges:
                if self.edge_attribute_mapper:
                    edge_data = self.edge_attribute_mapper.get_attributes(
                        graph.nodes[src_node], graph.nodes[target_node]
                    )
                else:
                    edge_data = {}

                perturbed_graph = nx.Graph(graph)
                perturbed_graph.add_edge(src_node, target_node, **edge_data)

                self._log_perturbation(
                    "edge_addition", src_node, target_node, edge_data
                )
                self.edit_count += 1

                return perturbed_graph
            else:
                resamples += 1

        raise NoValidEdgeError(f"No valid edge found after {max_resamples} tries")


class EdgeDeletionPerturbation(BasePerturbation):
    def __init__(self) -> None:
        """Applies a perturbation to the graph by deleting an edge"""
        super().__init__()

    def create(self, graph: nx.Graph) -> nx.Graph:
        nodes = list(graph.nodes)
        random.shuffle(nodes)

        # Searches for a node with at least 2 edges
        node_idx = 0
        while node_idx < len(nodes):
            node = nodes[node_idx]
            if len(graph.edges(node)) > 1:
                neighbors = list(graph.neighbors(node))
                random.shuffle(neighbors)

                # Ensures neighbor has at least 2 edges
                neighbor_idx = 0
                while neighbor_idx < len(neighbors):
                    neighbor = neighbors[neighbor_idx]
                    if len(graph.edges(neighbor)) > 1:
                        perturbed_graph = nx.Graph(graph)
                        perturbed_graph.remove_edge(node, neighbor)

                        self._log_perturbation(
                            "edge_deletion", src_node=node, target_node=neighbor
                        )
                        self.edit_count += 1

                        return perturbed_graph

                    neighbor_idx += 1
            node_idx += 1

        raise NoValidEdgeError("No pair of nodes found with at least 1 other edge each")


class EdgeReplacementPerturbation(BasePerturbation):
    def __init__(
        self,
        node_name_id: str,
        edge_name_id: str,
        replace_map: dict[str, list[str]],
        edge_attribute_mapper: EdgeAttributeMapper,
    ) -> None:
        """Applies a perturbation to the graph by replacing an edge

        Parameters
        ----------
        node_name_id : str
            Identifier to use as the node name
        edge_name_id : str
            Identifier to use as the edge name
        replace_map : dict[str, list[str]]
            Mapping of edge names to possible replacement values
        edge_attribute_mapper : EdgeAttributeMapper
            Object to use for generating edge attributes
        """
        super().__init__()
        self.node_name_id = node_name_id
        self.edge_name_id = edge_name_id
        self.replace_map = replace_map
        self.edge_attribute_mapper = edge_attribute_mapper

        self.memory = set()

    def create(self, graph: nx.Graph) -> nx.Graph:
        edges = list(graph.edges)

        # shuffle to avoid bias
        random.shuffle(edges)

        for edge in graph.edges:
            src_node = graph.nodes[edge[0]]
            target_node = graph.nodes[edge[1]]
            edge_name = (
                f"{src_node[self.node_name_id]}_{target_node[self.node_name_id]}"
            )

            if edge_name in self.replace_map and edge not in self.memory:
                current_value = graph[edge[0]][edge[1]][self.edge_name_id]

                alternative_values = [
                    v for v in self.replace_map[edge_name] if v != current_value
                ]
                new_value = random.choice(alternative_values)

                edge_data = self.edge_attribute_mapper.get_attributes(
                    src_node, target_node, edge_value=new_value
                )

                perturbed_graph = nx.Graph(graph)

                # Remove the old edge and replace with a new one
                perturbed_graph.remove_edge(edge[0], edge[1])
                perturbed_graph.add_edge(edge[0], edge[1], **edge_data)

                self._log_perturbation(
                    "edge_replacement",
                    src_node=edge[0],
                    target_node=edge[1],
                    metadata=edge_data,
                )
                self.edit_count += 1
                self.memory.add(edge)

                return perturbed_graph

        raise NoValidEdgeError("No valid replacement found based on the replace map")


class NodeRemovalPerturbation(BasePerturbation):
    def __init__(self) -> None:
        super().__init__()

    def create(self, graph: nx.Graph) -> nx.Graph:
        node = random.choice(list(graph.nodes))

        node_edges = graph.degree[node]  # type: ignore

        perturbed_graph = nx.Graph(graph)
        perturbed_graph.remove_node(node)
        self._log_perturbation(
            perturbation_type="node_removal",
            src_node=node,
            target_node=None,
            metadata=graph.nodes[node],
        )
        self.edit_count += 1 + node_edges

        return graph


class GraphPerturber:
    def __init__(
        self,
        perturbations: list[BasePerturbation],
        p_prob: Optional[np.ndarray | list[float]] = None,
    ) -> None:
        """Applies perturbations to a graph

        Class applies random perturbations to a graph and returns the perturbed
        graph. Perturbation operation options are defined as
        `BasePerturbation` objects

        Parameters
        ----------
        perturbations : list[BasePerturbation]
            A list of BasePerturbation objects representing the perturbations
            operations to make to the graph.
        p_prob : Optional[np.ndarray | list[float]]
            The probability distribution for selecting perturbation operations.
            If not provided, a uniform distribution is used by default.
            The sum of probabilities must be equal to 1. Defaults to None (a
            uniform sample probability)

        Raises
        ------
        ValueError
            If the sum of probabilities in `p_prob` is not equal to 1.
        """
        self.perturbations = perturbations
        self.perturbation_log = []
        self.total_edits = 0
        self.p_prob = p_prob

        if not self.p_prob:
            self.p_prob = np.array([1 / len(perturbations)] * len(perturbations))

        if isinstance(self.p_prob, list):
            self.p_prob = np.array(self.p_prob)

        if len(self.p_prob) != len(self.perturbations):
            raise ValueError(
                f"Length of `p_prob` is {len(self.p_prob)} which does not "
                f"match the number of perturbations: {len(self.perturbations)}"
            )

        if not np.isclose(self.p_prob.sum(), 1.0):
            raise ValueError("`p_prob` must sum to 1")

    def perturb(
        self, graph: nx.Graph, n_perturbations: int, max_retries: Optional[int] = None
    ) -> nx.Graph:
        p_graph = nx.Graph(graph)

        if not max_retries:
            max_retries = 3 * n_perturbations

        p_count = 0
        retries = 0
        while p_count < n_perturbations and retries < max_retries:
            sample_idx = np.random.choice(
                np.arange(len(self.perturbations)), p=self.p_prob
            )
            perturber = self.perturbations[sample_idx]
            try:
                p_graph = perturber.create(p_graph)
                self.perturbation_log.append(perturber.perturbation_log[-1])
                self.total_edits += perturber.edit_count
                p_count += 1
            except NoValidEdgeError:
                retries += 1
                continue

        if p_count == n_perturbations:
            return p_graph
        else:
            raise NoValidEdgeError(
                f"Could not find enough valid edges after {max_retries} retries"
            )
