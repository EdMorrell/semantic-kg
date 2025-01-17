import argparse
from pathlib import Path
import random

from dotenv import load_dotenv
import numpy as np

from semantic_kg.datasets import oregano
from semantic_kg.sampling import SubgraphDataset, SubgraphSampler


ROOT_DIR = Path(__file__).parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oregano_data_dir",
        type=Path,
        help="Path to data directory containing Oregano files",
        default=ROOT_DIR / "datasets" / "oregano",
    )
    parser.add_argument(
        "--triple_fpath",
        type=Path,
        help="Path to tsv file containing triple data",
        default=ROOT_DIR / "datasets" / "oregano" / "OREGANO_V2.1.tsv",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        help="Number of subgraphs to include in final dataset",
        default=3000,
    )
    parser.add_argument(
        "--max_neighbors",
        type=int,
        help="Maximum number of neighbors per node in subgraph",
        default=3,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Path to save final dataset",
        default="datasets/oregano/",
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
    oregano_data_dir: str | Path,
    triple_fpath: str | Path,
    n_iter: int,
    max_neighbors: int,
    save_path: Path,
    min_nodes: int = 3,
    max_nodes: int = 10,
    min_p_perturb: float = 0.1,
    max_p_perturb: float = 0.7,
) -> None:
    pass

    df = oregano.load_triple_df(str(triple_fpath))
    df = df.drop_duplicates()

    df = oregano.get_node_types(df)
    relation_map = oregano.create_relation_map(df)
    valid_edges = df["node_types"].unique().tolist()

    oregano_loader = oregano.OreganoLoader(oregano_data_dir)
    g = oregano_loader.load()

    perturber = oregano.build_oregano_perturber(
        relation_map=relation_map,
        valid_edges=valid_edges,
        replace_map=oregano.OREGANO_REPLACE_MAP,
    )

    sampler = SubgraphSampler(
        graph=g, node_index_field="node_id", method="bfs_node_diversity"
    )

    subgraph = SubgraphDataset(
        graph=g,
        subgraph_sampler=sampler,
        perturber=perturber,
        node_name_field="display_name",
        edge_name_field="predicate",
        n_node_range=(min_nodes, max_nodes),
        p_perturbation_range=(min_p_perturb, max_p_perturb),
        max_neighbors=max_neighbors,
        dataset_save_dir=save_path,
        save_subgraphs=False,
    )

    # NOTE: Oregano requires more retries to generate subgraphs
    subgraph.generate(n_iter, n_iter * 10)


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
        oregano_data_dir=args.oregano_data_dir,
        triple_fpath=args.triple_fpath,
        n_iter=args.n_iter,
        max_neighbors=args.max_neighbors,
        save_path=save_path,
        **kwargs,
    )
