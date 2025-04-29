import argparse
import yaml
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from bin import DATASET_CONFIG_PATHS, DATASET_LOADERS, SubgraphDatasetConfig
from semantic_kg.utils import (
    create_edge_map,
    create_replace_map,
    get_start_nodes_from_replace_map,
)
from semantic_kg.generation import SubgraphPipeline
from semantic_kg.perturbation import (
    GraphPerturber,
    EdgeDeletionPerturbation,
    EdgeReplacementPerturbation,
    NodeRemovalPerturbation,
    NodeReplacementPerturbation,
)


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


def _generate_node_type_replace_map(
    triple_df: pd.DataFrame,
    src_node_name_field: str,
    src_node_type_field: str,
    target_node_name_field: str,
    target_node_type_field: str,
) -> dict[str, list[str]]:
    src_node_type_dict = (
        triple_df.groupby(src_node_type_field)[src_node_name_field]
        .apply(set)
        .apply(list)
        .to_dict()
    )
    target_node_type_dict = (
        triple_df.groupby(target_node_type_field)[target_node_name_field]
        .apply(set)
        .apply(list)
        .to_dict()
    )
    return {
        key: src_node_type_dict.get(key, []) + target_node_type_dict.get(key, [])
        for key in set(
            list(src_node_type_dict.keys()) + list(target_node_type_dict.keys())
        )
    }


def _generate_dataset(
    config: SubgraphDatasetConfig,
    perturber: GraphPerturber,
    df: pd.DataFrame,
    n_iter: int,
    min_nodes: int,
    max_nodes: int,
    save_path: Path,
    p_perturbation_range: Optional[tuple[float, float]] = None,
    start_node_types: Optional[list[str]] = None,
    start_node_names: Optional[list[str]] = None,
) -> pd.DataFrame:
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
        perturber=perturber,
        p_perturbation_range=p_perturbation_range,
    )
    return pipeline.generate(
        df,
        n_iter=n_iter,
        n_node_range=(min_nodes, max_nodes),
        start_node_types=start_node_types,
        start_node_names=start_node_names,
        max_retries=n_iter
        * 1000,  # Needs to be really high as some perturbations only valid in rare instances  # noqa: E501
    )


def main(
    config: SubgraphDatasetConfig,
    n_iter: int,
    random_seed: int,
    save_path: Path,
    min_nodes: int = 10,
    max_nodes: int = 20,
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

    edge_deletion_perturbation = EdgeDeletionPerturbation(
        directed=config.directed_graph
    )
    edge_deletion_perturber = GraphPerturber(
        perturbations=[edge_deletion_perturbation],
        node_id_field="node_name",
        edge_id_field="edge_name",
    )

    if config.replace_map is None:
        if config.edge_map is None:
            edge_map = create_edge_map(
                df,
                config.src_node_type_field,
                config.target_node_type_field,
                config.edge_name_field,
                config.directed_graph,
            )
        replace_map = create_replace_map(edge_map)
    else:
        replace_map = config.replace_map

    replace_start_nodes = get_start_nodes_from_replace_map(
        replace_map=replace_map,
        triple_df=df,
        edge_name_field=config.edge_name_field,
        attr_field=config.src_node_name_field,
    )

    edge_replacement_perturbation = EdgeReplacementPerturbation(
        replace_map=replace_map,
        directed=config.directed_graph,
        node_type_field="node_type",
        edge_name_field="edge_name",
    )
    edge_replacement_perturber = GraphPerturber(
        perturbations=[edge_replacement_perturbation],
        node_id_field="node_name",
        edge_id_field="edge_name",
    )

    node_removal_perturbation = NodeRemovalPerturbation(directed=config.directed_graph)
    node_removal_perturber = GraphPerturber(
        perturbations=[node_removal_perturbation],
        node_id_field="node_name",
        edge_id_field="edge_name",
    )

    node_type_replace_map = _generate_node_type_replace_map(
        df,
        src_node_name_field=config.src_node_name_field,
        src_node_type_field=config.src_node_type_field,
        target_node_name_field=config.target_node_name_field,
        target_node_type_field=config.target_node_type_field,
    )
    node_replacement_perturbation = NodeReplacementPerturbation(
        node_attr_field="node_name", replace_map={"node_type": node_type_replace_map}
    )
    node_replacement_perturber = GraphPerturber(
        perturbations=[node_replacement_perturbation],
        node_id_field="node_name",
        edge_id_field="edge_name",
    )

    perturbers = {
        "edge_deletion": edge_deletion_perturber,
        "edge_replacement": edge_replacement_perturber,
        "node_removal": node_removal_perturber,
        "node_replacement": node_replacement_perturber,
    }

    dfs = []
    for p_type, perturber in perturbers.items():
        if p_type == "edge_replacement":
            kwargs = {
                "p_perturbation_range": (0.01, 0.1),
                "start_node_names": replace_start_nodes,
            }
        else:
            kwargs = {}
        print(f"Generating dataset for {p_type.replace("_", " ")} perturbation")
        p_df = _generate_dataset(
            config,
            perturber,
            df,
            n_iter,
            min_nodes,
            max_nodes,
            save_path,
            **kwargs,
        )
        p_df["perturbation_type"] = p_type
        dfs.append(p_df)

    df = pd.concat(dfs)

    fpath = save_path / "multisubgraph_dataset.csv"
    print(f"Final dataset saved to {fpath}")
    df.to_csv(fpath, index=False)


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
