import random

import pandas as pd


def load_triple_df(node_fpath: str, edge_fpath: str) -> pd.DataFrame:
    node_df = pd.read_csv(node_fpath)
    edge_df = pd.read_csv(edge_fpath)

    triple_df = edge_df.join(node_df, on="x_index", rsuffix="x").join(
        node_df, on="y_index", rsuffix="_y"
    )

    triple_df = triple_df.rename(
        {
            "node_index": "src_node_index",
            "node_type": "src_node_type",
            "node_name": "src_node_name",
            "node_index_y": "target_node_index",
            "node_type_y": "target_node_type",
            "node_name_y": "target_node_name",
        },
        axis=1,
    )

    return triple_df[
        [
            "src_node_index",
            "src_node_type",
            "src_node_name",
            "display_relation",
            "target_node_index",
            "target_node_type",
            "target_node_name",
        ]
    ]


if __name__ == "__main__":
    import numpy as np

    from semantic_kg.sampling import SubgraphSampler, SubgraphDataset
    from semantic_kg.perturbation import build_perturber
    from semantic_kg.datasets import KGLoader, create_edge_map, get_valid_node_pairs

    random.seed(42)
    np.random.seed(42)

    node_fpath = "datasets/prime_kg/nodes.csv"
    edge_fpath = "datasets/prime_kg/edges.csv"

    df = load_triple_df(node_fpath, edge_fpath)

    kg_loader = KGLoader(
        src_node_id_field="src_node_index",
        src_node_type_field="src_node_type",
        src_node_name_field="src_node_name",
        edge_name_field="display_relation",
        target_node_id_field="target_node_index",
        target_node_type_field="target_node_type",
        target_node_name_field="target_node_name",
    )
    g = kg_loader.load(triple_df=df, directed=False)

    edge_map = create_edge_map(
        df,
        src_node_type_field="src_node_type",
        target_node_type_field="target_node_type",
        edge_name_field="display_relation",
    )
    replace_map = {k: v for k, v in edge_map.items() if len(v) > 1}
    valid_node_pairs = get_valid_node_pairs(
        df,
        src_node_type_field="src_node_type",
        target_node_type_field="target_node_type",
    )

    perturber = build_perturber(
        edge_map=edge_map,
        valid_node_pairs=valid_node_pairs,  # type: ignore
        replace_map=replace_map,
        directed=False,
        p_prob=[0.3, 0.3, 0.3, 0.1],
    )

    sampler = SubgraphSampler(graph=g, method="bfs_node_diversity")

    subgraph = SubgraphDataset(
        graph=g,
        subgraph_sampler=sampler,
        perturber=perturber,
        n_node_range=(3, 10),
        save_subgraphs=False,
    )
    sample_df = subgraph.generate(1000, 10000)
