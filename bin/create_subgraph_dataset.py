import yaml
import random
import argparse
from pathlib import Path
from typing import Optional
from typing_extensions import Annotated

import numpy as np
from pydantic import BaseModel, AfterValidator

from semantic_kg.datasets import EDGE_MAPPING_TYPE
from semantic_kg.generation import SubgraphPipeline
from semantic_kg.datasets.oregano import OreganoLoader
from semantic_kg.datasets.prime_kg import PrimeKGLoader


ROOT_DIR = Path(__file__).parent.parent


CONFIG_PATHS = {
    "oregano": ROOT_DIR / "config" / "datasets" / "oregano.yaml",
    "prime_kg": ROOT_DIR / "config" / "datasets" / "prime_kg.yaml",
}


DATASET_LOADERS = {
    "oregano": OreganoLoader,
    "prime_kg": PrimeKGLoader,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        help=f"Name of dataset to load. Options are {list(CONFIG_PATHS.keys())}",
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


def _dataset_name_validator(name: str) -> str:
    if name not in DATASET_LOADERS:
        raise ValueError(
            f"Invalid dataset {name}. Valid datasets are {list(DATASET_LOADERS.keys())}"
        )

    return name


def _dir_exists_validator(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    if not data_dir:
        raise NotADirectoryError(f"{data_dir} does not exist")

    return data_dir


class SubgraphDatasetConfig(BaseModel):
    dataset_name: Annotated[str, AfterValidator(_dataset_name_validator)]
    data_dir: Annotated[str | Path, AfterValidator(_dir_exists_validator)]
    src_node_id_field: str
    src_node_type_field: str
    src_node_name_field: str
    edge_name_field: str
    target_node_id_field: str
    target_node_type_field: str
    target_node_name_field: str
    directed_graph: bool

    edge_map: Optional[EDGE_MAPPING_TYPE] = None
    replace_map: Optional[EDGE_MAPPING_TYPE] = None
    valid_node_pairs: Optional[list[tuple[str, str]]] = None


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
        config_path = CONFIG_PATHS[args.dataset_name]
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
