import random
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from semantic_kg.datasets import prime_kg
from semantic_kg.sampling import SubgraphSampler, SubgraphDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--node_dpath",
        type=str,
        help="Path to PrimeKG nodes CSV file",
        # TODO: Make path relative to repo
        default="~/Downloads/dataverse_files/nodes.csv",
    )
    parser.add_argument(
        "--edge_dpath",
        type=str,
        help="Path to PrimeKG edges CSV file",
        default="~/Downloads/dataverse_files/edges.csv",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        help="Number of subgraphs to include in final dataset",
        default=500,
    )
    parser.add_argument(
        "--max_neighbors",
        type=int,
        help="Maximum number of neighbors per node in subgraph",
        default=3,
    )
    parser.add_argument(
        "--start_node_types", nargs="*", type=str, default=["drug", "disease"]
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Path to save final dataset",
        default="datasets/prime_kg/",
    )
    parser.add_argument(
        "--min_nodes",
        type=int,
        help="Minimum number of nodes per subgraph",
        required=False,
    )
    parser.add_argument(
        "--max_nodes",
        type=int,
        help="Maximum number of nodes per subgraph",
        required=False,
    )
    parser.add_argument(
        "--min_p_perturb",
        type=float,
        help="Minimum number of perturbations expressed as a proportion of total nodes",
        required=False,
    )
    parser.add_argument(
        "--max_p_perturb",
        type=float,
        help="Maximum number of perturbations expressed as a proportion of total nodes",
        required=False,
    )
    parser.add_argument(
        "--random_seed", type=int, help="Random seed for generation", default=42
    )

    return parser.parse_args()


def main(
    node_dpath: str | Path,
    edge_dpath: str | Path,
    n_iter: int,
    max_neighbors: int,
    save_path: Path,
    min_nodes: int = 3,
    max_nodes: int = 12,
    min_p_perturb: float = 0.1,
    max_p_perturb: float = 0.7,
    start_node_types: Optional[list[str]] = None,
) -> None:
    """Generates a dataset of subgraphs and perturbed subgraphs"""
    node_df = pd.read_csv(node_dpath)
    edge_df = pd.read_csv(edge_dpath)

    relation_map = prime_kg.create_relation_map(node_df, edge_df)
    display_relation_map = prime_kg.create_display_relation_map(edge_df)

    # Any relations with multiple allowed values added to `replace_map`
    replace_map = {k: v for k, v in relation_map.items() if len(v) > 1}

    prime_kg_loader = prime_kg.PrimeKGLoader(node_df=node_df, edge_df=edge_df)
    graph = prime_kg_loader.load()

    # Creates the sub-graph perturber object
    perturber = prime_kg.build_primekg_perturber(
        relation_map, display_relation_map, replace_map
    )
    subgraph_sampler = SubgraphSampler(
        graph,
        node_index_field="node_index",
        method="bfs_node_diversity",
    )

    # Only allows start nodes of a given type
    if start_node_types:
        start_node_attrs = {"node_type": nt for nt in start_node_types}
    else:
        start_node_attrs = None

    subgraph_dataset = SubgraphDataset(
        graph=graph,
        subgraph_sampler=subgraph_sampler,
        perturber=perturber,
        node_name_field="node_name",
        edge_name_field="display_relation",
        n_node_range=(min_nodes, max_nodes),
        p_perturbation_range=(min_p_perturb, max_p_perturb),
        max_neighbors=max_neighbors,
        start_node_attrs=start_node_attrs,
        dataset_save_dir=save_path,
    )
    subgraph_dataset.generate(n_iter=n_iter)


if __name__ == "__main__":
    load_dotenv()
    args = parse_args()

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    kwargs = {}
    if args.min_nodes is not None:
        kwargs["min_nodes"] = args.min_nodes
    if args.max_nodes is not None:
        kwargs["max_nodes"] = args.max_nodes
    if args.min_p_perturb is not None:
        kwargs["min_p_perturb"] = args.min_p_perturb
    if args.max_p_perturb is not None:
        kwargs["max_p_perturb"] = args.max_p_perturb

    main(
        node_dpath=args.node_dpath,
        edge_dpath=args.edge_dpath,
        n_iter=args.n_iter,
        max_neighbors=args.max_neighbors,
        save_path=save_path,
        **kwargs,
    )
