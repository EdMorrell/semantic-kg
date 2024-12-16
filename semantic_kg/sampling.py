import pickle
import hashlib
from pathlib import Path
import random
from typing import Any, Literal, Optional
from collections import deque

import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm

from semantic_kg.perturbation import GraphPerturber, NoValidEdgeError


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
            # Convert to int
            edge = edge.item()

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
        node_index_field: str,
        method: Literal["bfs", "bfs_node_diversity"] = "bfs",
    ) -> None:
        """Class for sampling a subgraph from a graph

        Parameters
        ----------
        graph : nx.Graph
            Graph to sample from
        node_index_field : str
            Name of field to use as node index
        method : Literal["bfs", "bfs_node_diversity"], optional
            Search method to use for sampling subgraph, by default "bfs"

        Raises
        ------
        ValueError
            If search method is not valid
        """
        self.g = graph
        self.method = method
        self.node_index_field = node_index_field
        if method not in self.METHOD_MAP:
            raise ValueError(f"Invalid method: {method}")

    def sample(
        self,
        n_nodes: int,
        start_node: Optional[str | int] = None,
        max_neighbors: int = 5,
        **kwargs,
    ) -> nx.Graph:
        """Sample a subgraph from the graph.

        Parameters
        ----------
        n_nodes (int): Number of nodes to sample.
        start_node (Optional[str | int], optional): Starting node for sampling.
        Defaults to None.
        max_neighbors (int, optional): Maximum number of neighbors to consider for
        each node. Defaults to 5.
        **kwargs: Additional keyword arguments.

        Returns
        -------
            nx.Graph: Subgraph containing the sampled nodes and their edges.
        """
        if not start_node:
            start_node = random.choice(self.g.nodes)["node_index"]

        subgraph_edges = SubgraphSampler.METHOD_MAP[self.method](
            self.g, start_node, n_nodes, max_neighbors, **kwargs
        )
        nodes = [item for t in subgraph_edges for item in t]
        return self.g.subgraph(nodes)


def generate_triples(
    graph: nx.Graph,
    node_name_field: str,
    edge_name_field: str,
    node_attr_fields: Optional[list[str]] = None,
    edge_attr_fields: Optional[list[str]] = None,
) -> list[dict[str, str]]:
    """
    Generate triples from a graph.

    Parameters
    ----------
    graph : nx.Graph
        The input graph.
    node_name_field : str
        The name of the field in the graph nodes that represents the node name.
    edge_name_field : str
        The name of the field in the graph edges that represents the edge name.
    node_attr_fields : Optional[list[str]], optional
        The list of attribute fields to include in the node metadata, by default None.
    edge_attr_fields : Optional[list[str]], optional
        The list of attribute fields to include in the edge metadata, by default None.

    Returns
    -------
    list[dict[str, str]]
        A list of dictionaries representing the generated triples. Each dictionary
        contains the following keys:
        - 'source_node': A dictionary representing the source node, with the 'name' key
        representing the node name.
        - 'relation': A dictionary representing the edge, with the 'name' key
        representing the edge name.
        - 'target_node': A dictionary representing the target node, with the 'name' key
        representing the node name.
    """

    def _add_meta_to_dict(
        triplet_dict: dict[str, str], attr_fields: list[str], metadata: dict[str, Any]
    ) -> dict[str, str | dict]:
        """Add metadata to a dictionary."""
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


class SubgraphDataset:
    def __init__(
        self,
        graph: nx.Graph,
        subgraph_sampler: SubgraphSampler,
        perturber: GraphPerturber,
        node_name_field: str,
        edge_name_field: str,
        n_node_range: tuple[int, int] = (3, 12),
        p_perturbation_range: tuple[float, float] = (0.1, 0.7),
        max_neighbors: int = 3,
        start_node_attrs: Optional[dict[str, str]] = None,
        dataset_save_dir: Optional[Path] = None,
        save_subgraphs: bool = True,
        subgraph_save_dir: Optional[Path] = None,
    ) -> None:
        """Class to generate a dataset of subgraphs and perturbed subgraphs

        Parameters
        ----------
        graph : nx.Graph
            The input graph from which subgraphs will be sampled.
        subgraph_sampler : SubgraphSampler
            The subgraph sampler object used to sample subgraphs from the input graph.
        perturber : GraphPerturber
            The graph perturber object used to perturb the sampled subgraphs.
        node_name_field : str
            Name of node field to use in final dataset
        edge_name_field : str
            Name of edge field to use in final dataset
        n_node_range : tuple[int, int], optional
            The range of number of nodes for the sampled subgraphs. Default is (2, 10).
        p_perturbation_range : tuple[float, float], optional
            Range of fraction of total nodes to perturb. Default is (0.1, 0.7).
        max_neighbors : int, optional
            The maximum number of neighbors to consider when sampling subgraphs.
            Default is 3.
        start_node_attrs : Optional[dict[str, str]], optional
            The attributes of the start node for the sampled subgraphs.
            Default is None.
        dataset_save_dir : Optional[Path], optional
            The directory to save the final dataset. If not provided, a default
            directory will be used.
        save_subgraphs : bool, optional
            If True, then saves intermediate subgraph objects. Defaults to True.
            NOTE: Objects saved as pickle files making saving very slow
        subgraph_save_dir : Optional[Path], optional
            The directory to save each generated subgraph. If not provided, a default
            directory will be used.
        """
        self.graph = graph
        self.subgraph_sampler = subgraph_sampler
        self.perturber = perturber
        self.node_name_field = node_name_field
        self.edge_name_field = edge_name_field
        self.n_node_range = n_node_range
        self.p_perturbation_range = p_perturbation_range
        self.max_neighbors = max_neighbors
        self.start_node_attrs = start_node_attrs

        # Configure save directories
        root_dir = Path(__file__).parent.parent
        if not dataset_save_dir:
            dataset_save_dir = Path(root_dir) / "datasets"
            dataset_save_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_save_dir = dataset_save_dir

        self.save_subgraphs = save_subgraphs
        if not subgraph_save_dir and self.save_subgraphs:
            subgraph_save_dir = Path(root_dir) / "outputs"
            subgraph_save_dir.mkdir(parents=True, exist_ok=True)
        self.subgraph_save_dir = subgraph_save_dir

    def _sample_start_node(self) -> int | str:
        all_nodes = list(self.graph.nodes)
        if self.start_node_attrs:
            valid_nodes = [
                n
                for n in all_nodes
                if any(
                    [
                        self.graph.nodes[n][k] == v
                        for k, v in self.start_node_attrs.items()
                    ]
                )
            ]
            if not valid_nodes:
                raise ValueError(
                    f"No nodes found matching attributes: {self.start_node_attrs}"
                )
        else:
            valid_nodes = all_nodes

        return random.choice(valid_nodes)

    def _get_graph_hash(self, subgraph: nx.Graph) -> str:
        return nx.weisfeiler_lehman_graph_hash(
            subgraph, edge_attr=self.edge_name_field, node_attr=self.node_name_field
        )

    def _save_subgraph(self, subgraph: nx.Graph, save_path: Path) -> None:
        """Saves the subgraph under a unique hash"""
        if not save_path.is_file():
            with open(str(save_path), "wb") as f:
                pickle.dump(subgraph, f)

    def _save_subgraphs(
        self,
        subgraph: nx.Graph,
        perturbed_subgraph: nx.Graph,
        subgraph_hash: str,
        perturbed_subgraph_hash: str,
    ) -> None:
        """Save the subgraphs to the specified directory"""
        if self.subgraph_save_dir:  # Necessary to avoid type-error
            subgraph_save_path = self.subgraph_save_dir / f"{subgraph_hash}.pkl"
            perturbed_subgraph_save_path = (
                self.subgraph_save_dir / f"{perturbed_subgraph_hash}.pkl"
            )
            self._save_subgraph(subgraph, subgraph_save_path)
            self._save_subgraph(perturbed_subgraph, perturbed_subgraph_save_path)

    def _format_row_data(
        self,
        subgraph: nx.Graph,
        perturbed_subgraph: nx.Graph,
        perturbation_log: list[dict[str, Any]],
        subgraph_hash: str,
        perturbed_subgraph_hash: str,
    ) -> dict[str, Any]:
        """Formats data for a subgraph-pair into a dictionary"""
        subgraph_triples = generate_triples(
            subgraph,
            node_name_field=self.node_name_field,
            edge_name_field=self.edge_name_field,
        )
        perturbed_subgraph_triples = generate_triples(
            perturbed_subgraph,
            node_name_field=self.node_name_field,
            edge_name_field=self.edge_name_field,
        )

        n_nodes = subgraph.number_of_nodes()
        n_edges = subgraph.number_of_edges()
        total_edits = self.perturber.total_edits

        similarity = 1 - (total_edits / (n_nodes + n_edges))

        return {
            "subgraph_triples": subgraph_triples,
            "perturbed_subgraph_triples": perturbed_subgraph_triples,
            "perturbation_log": perturbation_log,
            "similarity": similarity,
            "subgraph_hash": subgraph_hash,
            "perturbed_subgraph_hash": perturbed_subgraph_hash,
        }

    def _save_dataset(self, subgraph_dataset: pd.DataFrame) -> None:
        """Saves the final dataet to a path based on a unique hash"""
        # Generates a unique hash for the dataset based on the hashes of individual
        # subgraphs and perturbed subgraphs
        hash_key = hashlib.md5()
        sg_hash_keys = subgraph_dataset["subgraph_hash"].to_list()
        psg_hash_keys = subgraph_dataset["perturbed_subgraph_hash"].to_list()
        all_keys = "".join(sg_hash_keys + psg_hash_keys)
        hash_key.update(all_keys.encode())
        save_hash = hash_key.hexdigest()

        save_path = self.dataset_save_dir / f"{save_hash}.csv"  # type: ignore
        subgraph_dataset.to_csv(save_path, index=False)

        print(f"Dataset saved to {save_path}")

    def generate(self, n_iter: int, max_retries: Optional[int] = None) -> pd.DataFrame:
        """Generates a dataset of subgraph/perturbed subgraph pairs

        Parameters
        ----------
        n_iter : int
            Number of iterations of generation. This will determined the
            number of rows in the final dataset
        max_retries: int, optional
            Number of perturbations to retry before giving up

        Returns
        -------
        pd.DataFrame
            Dataframe consisting of the following fields:
            - "subgraph_triples": List of triples representing the original
            subgraph
            - "perturbed_subgraph_triples": List of triples representing the
            perturbed subgraph
            - "similarity": Similarity score between original subgraph and perturbed
            subgraph
            - "subgraph_path": File path where the original subgraph is saved
            - "perturbed_subgraph_path": File path where the perturbed subgraph is
            saved
        """
        if not max_retries:
            max_retries = 2 * n_iter

        all_data = {}
        pbar = tqdm(total=n_iter)
        retries = 0
        idx = 0
        while idx < n_iter and retries < max_retries:
            # Reset number of edits
            self.perturber.reset()

            # Generate `n_nodes` within `n_node_range`
            n_nodes = random.randrange(self.n_node_range[0], self.n_node_range[1])

            # Generates a random fraction of total nodes to act as number of
            # perturbations
            n_perturbations = np.ceil(  # Used to avoid rounding down to 0
                random.uniform(
                    self.p_perturbation_range[0], self.p_perturbation_range[1]
                )
                * n_nodes
            ).item()

            start_node = self._sample_start_node()

            subgraph = self.subgraph_sampler.sample(
                n_nodes=n_nodes,
                start_node=start_node,
                max_neighbors=self.max_neighbors,
                node_type_field="node_type",
            )

            try:
                perturbed_subgraph = self.perturber.perturb(
                    subgraph, n_perturbations=n_perturbations
                )
            except NoValidEdgeError:
                retries += 1
                continue

            # Save both subgraphs and collect unique IDs
            subgraph_hash = self._get_graph_hash(subgraph)
            perturbed_subgraph_hash = self._get_graph_hash(perturbed_subgraph)

            if self.save_subgraphs:
                self._save_subgraphs(
                    subgraph, perturbed_subgraph, subgraph_hash, perturbed_subgraph_hash
                )

            row_data = self._format_row_data(
                subgraph,
                perturbed_subgraph,
                self.perturber.perturbation_log,
                subgraph_hash,
                perturbed_subgraph_hash,
            )

            all_data[idx] = row_data
            pbar.update(1)
            idx += 1

        if idx == n_iter:
            subgraph_dataset = pd.DataFrame(all_data).T
            self._save_dataset(subgraph_dataset)

            return subgraph_dataset
        else:
            raise ValueError(
                f"Max retries {max_retries} exceeded. "
                f"Unable to generate {n_iter} subgraphs."
                "Consider increasing `max_retries` or reducing `n_iter`"
            )
