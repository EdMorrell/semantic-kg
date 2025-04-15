import yaml
import random
import argparse
from pathlib import Path

import numpy as np

from bin import DATASET_CONFIG_PATHS, DATASET_LOADERS, SubgraphDatasetConfig
from semantic_kg.generation import SubgraphPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        help=f"Name of dataset to load. Options are {list(DATASET_CONFIG_PATHS.keys())}",
        required=False,
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        help="Optionally provide the path to a config directly",
        required=False,
    )
    parser.add_argument(
        "--random_seed", type=int, help="Random seed for generation", default=42
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        help="Number of subgraphs to include in final dataset",
        default=1000,
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
        "--save_path",
        type=Path,
        help="Path to save final dataset",
        default="datasets/",
    )

    return parser.parse_args()


def main(
    config: SubgraphDatasetConfig,
    n_iter: int,
    random_seed: int,
    save_path: Path,
    min_nodes: int = 3,
    max_nodes: int = 10,
) -> None:
    """Entry-point to subgraph generation pipeline code

    Parameters
    ----------
    config : SubgraphDatasetConfig
        Config describing knowledge-graph parameters
    n_iter : int
        Number of rows in final dataset
    random_seed : int
        Random state for generation
    save_path : Path
        Path to directory to save data
    min_nodes : int, optional
        Minimum number of nodes per subgraph, by default 3
    max_nodes : int, optional
        Maximum number of nodes per subgraph, by default 10
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    loader = DATASET_LOADERS[config.dataset_name](config.data_dir)
    df = loader.load()

    pipeline = SubgraphPipeline(
        src_node_id_field=config.src_node_id_field,
        src_node_type_field=config.src_node_type_field,
        src_node_name_field=config.src_node_name_field,
        edge_name_field=config.edge_name_field,
        target_node_id_field=config.target_node_id_field,
        target_node_type_field=config.target_node_type_field,
        target_node_name_field=config.target_node_name_field,
        save_path=save_path,
        directed=config.directed_graph,
        edge_map=config.edge_map,
        replace_map=config.replace_map,
        valid_node_pairs=config.valid_node_pairs,
    )
    pipeline.generate(df, n_iter=n_iter, n_node_range=(min_nodes, max_nodes))


if __name__ == "__main__":
    args = parse_args()

    if not args.dataset_name and not args.config_path:
        raise ValueError("`dataset_name` and `config_path` can't both be None")

    if args.dataset_name:
        config_path = DATASET_CONFIG_PATHS[args.dataset_name]
    else:
        config_path = args.config_path

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset_config = SubgraphDatasetConfig.model_validate(config)

    save_path = Path(args.save_path)
    if dataset_config.dataset_name not in str(save_path):
        save_path = save_path / dataset_config.dataset_name

    kwargs = {}
    if args.min_nodes is not None:
        kwargs["min_nodes"] = args.min_nodes
    if args.max_nodes is not None:
        kwargs["max_nodes"] = args.max_nodes

    main(
        dataset_config,
        n_iter=args.n_iter,
        random_seed=args.random_seed,
        save_path=save_path,
        **kwargs,
    )
