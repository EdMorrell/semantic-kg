import abc
import random
from typing import Optional, Protocol

import numpy as np
import networkx as nx

from semantic_kg import utils


class NoValidEdgeError(Exception):
    """Error for instance where no valid edges available for a perturbation"""

    pass


class NoValidNodeError(Exception):
    """Error for instance where no valid nodes available for a perturbation"""

    pass


class BasePerturbation(abc.ABC):
    def __init__(self, directed: bool = True) -> None:
        """Base class for a perturbation operation to apply to a graph"""
        self.directed = directed
        self.perturbation_log = []

        # Counts the edits made by a perturbation. Any edit is a single
        # change made to an edge or node.
        # NOTE: Node perturbations typically constitute >1 edit due to
        # their impact on surrounding edges
        self.edit_count = 0

    def reset(self) -> None:
        self.perturbation_log = []
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
        valid_node_pairs: Optional[list[tuple[str, str]]] = None,
        edge_map: Optional[dict[tuple[str, str], list[str]]] = None,
        directed: bool = True,
        temperature: float = 0.2,
        node_type_field: str = "node_type",
    ) -> None:
        """Applies a perturbation to the graph by adding an edge

        Parameters
        ----------
        valid_node_pairs : Optional[list[tuple[str, str]]], optional
            A list of tuples denoting node-type pairs for which an edge is allowed,
            If not provided then edges allowed between all node-type. By default None
        edge_map : Optional[dict[tuple[str, str], list[str]]], optional
            A map from node-type pairs to a list of valid edge-names. An edge-name will
            be randomly sampled from the list. If not provided then the edge will
            not be assigned a name, by default None
        directed : bool, optional
            Boolean indicating whether the graph is directed, by default True
        temperature : float, optional
            The create method uses a softmax function to sample the
            nodes to connect to based on the distance to that edge.
            The temperature parameter controls the randomness of the
            sampling, by default 0.2, which biases sampling towards
            closer nodes. To allow edges between more distant nodes,
            increase the temperature.
        node_type_field : str, optional
            Node attribute key denoting the type of the node, by default "node_type"
        """
        super().__init__(directed=directed)
        self.valid_node_pairs = valid_node_pairs
        self.edge_map = edge_map
        self.temperature = temperature
        self.node_type_field = node_type_field

    def create(self, graph: nx.Graph, max_resamples: int = 10) -> nx.Graph:
        all_distance = nx.floyd_warshall(graph)

        # Only sample from nodes with at least one edge greater than 1
        valid_nodes = [
            node
            for node in all_distance
            if any(v > 1 for v in all_distance[node].values())
        ]
        src_node = random.choice(valid_nodes)

        node_distances = {k: v for k, v in all_distance[src_node].items() if v > 1}

        # Raise an error if no edges greater than 1
        if not node_distances:
            raise NoValidEdgeError("No edge distances greater than 1.")

        distances = np.array(list(node_distances.values()))

        inv_distances = 1 / distances

        pvals = utils.softmax(inv_distances, self.temperature)

        resamples = 0
        while resamples < max_resamples:
            target_node = list(node_distances.keys())[
                np.where(np.random.multinomial(1, pvals))[0][0]
            ]

            src_node_type = graph.nodes[src_node][self.node_type_field]
            target_node_type = graph.nodes[target_node][self.node_type_field]

            node_pair = (src_node_type, target_node_type)

            if not self.directed:
                node_pair = sorted(node_pair)

            # Ensures only a valid pair is found
            if not self.valid_node_pairs or node_pair in self.valid_node_pairs:
                if self.edge_map:
                    if node_pair not in self.edge_map:
                        raise NoValidEdgeError(f"{node_pair=} not found in `edge_map`")

                    edge_opts = self.edge_map[node_pair]
                    edge_data = {"edge_name": random.choice(edge_opts)}
                else:
                    edge_data = {}

                # Checks if graph is a directed graph
                if isinstance(graph, nx.DiGraph):
                    perturbed_graph = nx.DiGraph(graph)
                else:
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
    def __init__(self, directed: bool = True) -> None:
        """Applies a perturbation to the graph by deleting an edge"""
        super().__init__(directed=directed)

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
                        # Checks if graph is a directed graph
                        if self.directed:
                            perturbed_graph = nx.DiGraph(graph)
                        else:
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
        replace_map: dict[tuple[str, str], list[str]],
        directed: bool = True,
        node_type_field: str = "node_type",
        edge_name_field: str = "edge_name",
    ) -> None:
        """Applies a perturbation to the graph by replacing an edge

        Parameters
        ----------
        replace_map : dict[tuple[str, str], list[str]]
            A map from node-type pairs to a list of valid edge-names. An edge-name will
            be randomly sampled from the list. If not provided then the edge will
            not be assigned a name, by default None
        directed : bool, optional
            Boolean indicating whether the graph is directed, by default True
        node_type_field : str, optional
            Name of node type field in graph, by default "node_type"
        edge_name_field : str, optional
            Name of edge name field in graph, by default "edge_name"
        """
        super().__init__(directed=directed)
        self.replace_map = replace_map
        self.node_type_field = node_type_field
        self.edge_name_field = edge_name_field

        self.memory = set()

    def create(self, graph: nx.Graph) -> nx.Graph:
        edges = list(graph.edges)

        # shuffle to avoid bias
        random.shuffle(edges)

        for edge in graph.edges:
            src_node = graph.nodes[edge[0]]
            target_node = graph.nodes[edge[1]]

            node_pair = (
                src_node[self.node_type_field],
                target_node[self.node_type_field],
            )
            if not self.directed:
                node_pair = tuple(sorted(node_pair))

            if node_pair in self.replace_map and edge not in self.memory:
                current_value = graph[edge[0]][edge[1]][self.edge_name_field]

                if current_value not in self.replace_map[node_pair]:
                    raise NoValidEdgeError(
                        "`replace_map` does not contain an alternative value"
                        f" for {current_value}"
                    )

                alternative_values = [
                    v for v in self.replace_map[node_pair] if v != current_value
                ]
                new_value = random.choice(alternative_values)

                edge_data = {self.edge_name_field: new_value}

                # Checks if graph is a directed graph
                if self.directed:
                    perturbed_graph = nx.DiGraph(graph)
                else:
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
    def __init__(self, directed: bool = True) -> None:
        """Class for removing a node from the graph"""
        super().__init__(directed=directed)

    def _check_is_star_center(self, graph: nx.Graph, node: str) -> bool:
        """Checks if the node is the center of a star graph"""
        return (
            all(graph.degree[n] == 1 for n in graph.nodes if n != node)  # type: ignore
            and graph.degree[node] == len(graph.nodes) - 1  # type: ignore
        )  # type: ignore

    def _get_edit_count(self, graph: nx.Graph, p_graph: nx.Graph, node: str) -> int:
        # The number of edits should be equal to:
        # 1 (node removal) + number of edges connected to the node
        # + any nodes that are isolated (degree 0) after removal
        isolated_nodes = len(
            [
                node
                for node in p_graph.nodes
                if p_graph.degree[node] == 0  # type: ignore
            ]
        )
        return 1 + graph.degree[node] + isolated_nodes  # type: ignore

    def _clean_isolated_nodes(self, p_graph: nx.Graph) -> nx.Graph:
        """Removes any isolated nodes from the graph"""
        isolated_nodes = [
            node
            for node in p_graph.nodes
            if p_graph.degree[node] == 0  # type: ignore
        ]

        if isinstance(p_graph, nx.DiGraph):
            perturbed_graph = nx.DiGraph(p_graph)
        else:
            perturbed_graph = nx.Graph(p_graph)

        perturbed_graph.remove_nodes_from(isolated_nodes)

        return perturbed_graph

    def create(self, graph: nx.Graph, rm_node: Optional[str] = None) -> nx.Graph:
        """Removes a node from the graph

        Method will remove a node from the graph and any corresponding edges.
        If any nodes are isolated after the removal, they will also be removed.

        The method will avoid removing nodes that are the center of a star graph
        as this would result in a fully disconnected graph.

        Parameters
        ----------
        graph : nx.Graph
            Graph to perturb
        rm_node : Optional[str], optional
            Optionally specify a node to remove (for testing purposes), by default None

        Returns
        -------
        nx.Graph
            Perturbed graph

        Raises
        ------
        NoValidNodeError
            If the total number of nodes is less than 3
        NoValidNodeError
            If the node to be removed is the center of a star graph
        """
        if len(graph.nodes) < 3:
            raise NoValidNodeError("Not enough nodes to remove one")

        if not rm_node:
            valid_nodes = [
                n for n in graph.nodes if not self._check_is_star_center(graph, n)
            ]
            node = random.choice(valid_nodes)
        else:
            node = rm_node
            if self._check_is_star_center(graph, node):
                raise NoValidNodeError("Node is the center of a star graph")

        # Checks if graph is a directed graph
        if self.directed:
            perturbed_graph = nx.DiGraph(graph)
        else:
            perturbed_graph = nx.Graph(graph)

        perturbed_graph.remove_node(node)
        self.edit_count += self._get_edit_count(graph, perturbed_graph, node)
        perturbed_graph = self._clean_isolated_nodes(perturbed_graph)

        self._log_perturbation(
            perturbation_type="node_removal",
            src_node=node,
            target_node=None,
            metadata=graph.nodes[node],
        )

        return perturbed_graph


class NodeReplacementPerturbation(BasePerturbation):
    def __init__(
        self,
        node_attr_field: str,
        replace_opts: Optional[list[str]] = None,
        replace_map: Optional[dict[str, dict[str, list[str]]]] = None,
        directed: bool = True,
    ) -> None:
        """Perturber for replacing a node

        This perturber will randomly replace a node's attributes, for example the
        node's name. What the attribute is replaced with can be defined as a list
        of possible options, or a map if you want to replace depending on another
        node attribute, for example, if the new attribute depends on the node's type

        If specifying a list of options to replace with, use: `replace_opts`
        whereas if specifying a map, use: `replace_map`

        Parameters
        ----------
        node_attr_field : str
            Node attribute field to replace
        replace_opts : Optional[list[str]], optional
            If all nodes can be replaced with any value, then specify a list of values
            the `node_attr_field` in the perturbed graph can take. If this arg is not
            specified then `replace_map` must be.
        replace_map : Optional[dict[str, dict[str, list[str]]]], optional
            A dict that defines options what new node attribute a node can
            take, based on the value of another attribute, for example:

            replace_map = {
                "node_type":
                    "A": [1, 2, 3],
                    "B": [4, 5, 6],
            }

            This will replace the `node_attr_field` with 1, 2, or 3 if the `node_type`
            attribute is set to "A", or 4, 5, or 6, if the `node_type` is set to "B".
        directed : bool, optional
            Whether the graph is directed or now, by default True

        Raises
        ------
        ValueError
            If neither `replace_opts` or `replace_map` is set, or if both are set.
        ValueError
            If more than one conditioning attribute is set in `replace_map`
        """
        super().__init__(directed)
        self.node_attr_field = node_attr_field
        if not replace_opts and not replace_map:
            raise ValueError("You must specify `replace_opts` or `replace_map`")
        if replace_opts and replace_map:
            raise ValueError("Cannot specify both `replace_map` and `replace_opts`")
        self.replace_opts = replace_opts
        self.replace_map = replace_map

        if self.replace_map:
            if len(self.replace_map.keys()) > 1:
                raise ValueError(
                    "Use of `replace_map` only supports conditioning on a "
                    "single node attribute"
                )

    def create(self, graph: nx.Graph, replace_node: Optional[str] = None) -> nx.Graph:
        """Replaces an attribute in a node

        Parameters
        ----------
        graph : nx.Graph
            Graph to perturb
        rm_node : Optional[str], optional
            Optionally specify a node to replace (for testing purposes), by default None

        Returns
        -------
        nx.Graph
            Perturbed graph

        Raises
        ------
        NoValidNodeError
            If there is no node to replace with found in the `replace_map`
        """
        if not replace_node:
            replace_node = np.random.choice(list(graph.nodes))

        node_attr_val = graph.nodes[replace_node][self.node_attr_field]
        if self.replace_map:
            replace_key = list(self.replace_map.keys())[0]
            replace_node_attr = graph.nodes[replace_node][replace_key]
            replace_opts = self.replace_map[replace_key][replace_node_attr]
        else:
            replace_opts = self.replace_opts

        valid_replace_opts = [
            attr
            for attr in replace_opts
            if attr != node_attr_val  # type: ignore
        ]
        if not valid_replace_opts:
            raise NoValidNodeError(
                f"No valid options found to replace {self.node_attr_field} "
                f"for node: {replace_node}"
            )

        p_graph = graph.copy()
        new_node = np.random.choice(valid_replace_opts)
        attr_update = {replace_node: {self.node_attr_field: str(new_node)}}
        nx.set_node_attributes(p_graph, attr_update)

        self._log_perturbation(
            perturbation_type="node_replacement",
            src_node=replace_node,
            target_node=None,
            metadata=attr_update,
        )

        return p_graph


class GraphPerturber:
    def __init__(
        self,
        perturbations: list[BasePerturbation],
        node_id_field: str,
        edge_id_field: str,
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
        node_id_field : str
            The field name to use as the node identifier when computing edit distance
        edge_id_field : str
            The field name to use as the edge identifier when computing edit distance
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
        self.node_id_field = node_id_field
        self.edge_id_field = edge_id_field
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

    def reset(self) -> None:
        """Resets the perturbation logs. Used when re-applying class to new graphs"""
        self.perturbation_log = []
        self.total_edits = 0

    def _apply_perturbation(
        self, graph: nx.Graph, perturber: BasePerturbation
    ) -> nx.Graph:
        p_graph = perturber.create(graph)
        perturbation_src = perturber.perturbation_log[-1]["source"]
        perturbation_target = perturber.perturbation_log[-1]["target"]

        # Check there isn't already a perturbation of these 2 nodes
        for perturbation in self.perturbation_log:
            if (
                perturbation["source"] == perturbation_src
                and perturbation["target"] == perturbation_target
            ) or (
                perturbation["source"] == perturbation_target
                and perturbation["target"] == perturbation_src
            ):
                raise NoValidEdgeError("Perturbation already exists")

        return p_graph

    def perturb(
        self, graph: nx.Graph, n_perturbations: int, max_retries: Optional[int] = None
    ) -> nx.Graph:
        # Checks if graph is a directed graph
        if isinstance(graph, nx.DiGraph):
            p_graph = nx.DiGraph(graph)
        else:
            p_graph = nx.Graph(graph)

        if not max_retries:
            max_retries = max(10, 3 * n_perturbations)

        p_count = 0
        retries = 0
        while p_count < n_perturbations and retries < max_retries:
            sample_idx = np.random.choice(
                np.arange(len(self.perturbations)), p=self.p_prob
            ).item()
            perturber = self.perturbations[sample_idx]

            # Sets edit history and count back to 0
            perturber.reset()

            try:
                p_graph = self._apply_perturbation(p_graph, perturber)
            except (NoValidEdgeError, NoValidNodeError, IndexError):
                retries += 1
                continue

            self.perturbation_log.append(perturber.perturbation_log[-1])
            p_count += 1

        if p_count == n_perturbations:
            edit_distance = utils.compute_edit_distance(
                graph,
                p_graph,
                node_id_field=self.node_id_field,
                edge_id_field=self.edge_id_field,
            )
            self.total_edits = edit_distance
            return p_graph
        else:
            raise NoValidEdgeError(
                f"Could not find enough valid edges after {max_retries} retries"
            )


def build_perturber(
    edge_map: dict[tuple[str, str], list[str]],
    valid_node_pairs: list[tuple[str, str]],
    replace_map: dict[tuple[str, str], list[str]],
    directed: bool,
    edge_addition: bool = True,
    edge_deletion: bool = True,
    edge_replacement: bool = True,
    node_removal: bool = True,
    p_prob: Optional[list[float]] = None,
    node_name_field: str = "node_name",
    edge_name_field: str = "edge_name",
) -> GraphPerturber:
    n_perturbers = sum([edge_addition, edge_deletion, edge_replacement, node_removal])
    if not p_prob:
        p_prob = [1 / n_perturbers] * n_perturbers

    if len(p_prob) != n_perturbers:
        raise ValueError(
            f"`p_prob` has size {len(p_prob)} which is not equal to the number of "
            f"perturbers: {n_perturbers}"
        )

    perturbers = []
    if edge_addition:
        perturbers.append(
            EdgeAdditionPerturbation(
                valid_node_pairs=valid_node_pairs,
                edge_map=edge_map,
                directed=directed,
            )
        )

    if edge_deletion:
        perturbers.append(EdgeDeletionPerturbation(directed=directed))

    if edge_replacement:
        perturbers.append(
            EdgeReplacementPerturbation(
                replace_map=replace_map,
                directed=directed,
            )
        )

    if node_removal:
        perturbers.append(NodeRemovalPerturbation(directed=directed))

    return GraphPerturber(
        perturbations=perturbers,
        node_id_field=node_name_field,
        edge_id_field=edge_name_field,
        p_prob=p_prob,
    )
