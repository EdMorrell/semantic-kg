import random
from typing import Optional

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import colormaps

from semantic_kg.perturbation import (
    EdgeAdditionPerturbation,
    EdgeDeletionPerturbation,
    EdgeReplacementPerturbation,
    NodeRemovalPerturbation,
    GraphPerturber,
)


def create_relation_map(
    node_df: pd.DataFrame, edge_df: pd.DataFrame
) -> dict[str, list[str]]:
    """Generates a mapping from node-type pairs to relation-types

    Uses all known node-type/node-type mappings in PrimeKG to find
    allowed node-type pair -> relation types
    """
    all_df = edge_df.join(node_df, on="x_index", rsuffix="x").join(
        node_df, on="y_index", rsuffix="_y"
    )
    all_df["node_types"] = all_df["node_type"] + "_" + all_df["node_type_y"]
    return (
        all_df[["node_types", "relation"]]
        .groupby("node_types")
        .agg(lambda x: list(x.unique()))["relation"]
        .to_dict()
    )


def create_display_relation_map(edge_df: pd.DataFrame) -> dict[str, list[str]]:
    """Creates a mapping from "relation" to "display_relation"""
    return (
        edge_df[["relation", "display_relation"]]
        .groupby("relation")
        .agg(lambda x: list(x.unique()))["display_relation"]
    ).to_dict()


class PrimeKGLoader:
    def __init__(self, node_df: pd.DataFrame, edge_df: pd.DataFrame) -> None:
        self.node_df = node_df
        self.edge_df = edge_df

    def load(self) -> nx.Graph:
        g = nx.from_pandas_edgelist(
            df=self.edge_df,
            source="x_index",
            target="y_index",
            edge_attr=["relation", "display_relation"],
        )
        node_dict = self.node_df.to_dict(orient="index")
        nx.set_node_attributes(g, node_dict)

        return g


class PrimeKGEdgeAttributeMapper:
    def __init__(
        self,
        relation_map: dict[str, list[str]],
        display_relation_map: dict[str, list[str]],
    ) -> None:
        self.relation_map = relation_map
        self.display_relation_map = display_relation_map

    def get_attributes(
        self,
        src_node: dict[str, str],
        target_node: dict[str, str],
        edge_value: Optional[str] = None,
    ) -> dict[str, str]:
        src_node_type = src_node["node_type"]
        target_node_type = target_node["node_type"]
        edge_name = f"{src_node_type}_{target_node_type}"

        if not edge_value:
            relation = random.choice(self.relation_map[edge_name])
        else:
            relation = edge_value

        display_relation = random.choice(self.display_relation_map[relation])

        return {
            "relation": relation,
            "display_relation": display_relation,
        }


def build_primekg_perturber(
    relation_map: dict[str, list[str]],
    display_relation_map: dict[str, list[str]],
    replace_map: dict[str, list[str]],
) -> GraphPerturber:
    edge_addition_perturbation = EdgeAdditionPerturbation(
        node_type_field="node_type",
        valid_node_pairs=list(relation_map.keys()),
        edge_attribute_mapper=PrimeKGEdgeAttributeMapper(
            relation_map=relation_map, display_relation_map=display_relation_map
        ),
    )
    edge_deletion_perturbation = EdgeDeletionPerturbation()
    edge_replacement_perturbation = EdgeReplacementPerturbation(
        node_type_field="node_type",
        edge_name_field="relation",
        replace_map=replace_map,
        edge_attribute_mapper=PrimeKGEdgeAttributeMapper(
            relation_map=relation_map,
            display_relation_map=display_relation_map,
        ),
    )
    node_removal_perturbation = NodeRemovalPerturbation()

    return GraphPerturber(
        perturbations=[
            edge_addition_perturbation,
            edge_deletion_perturbation,
            edge_replacement_perturbation,
            node_removal_perturbation,
        ],
        node_id_field="node_index",
        edge_id_field="display_relation",
        p_prob=[0.3, 0.3, 0.3, 0.1],
    )


def plot_primekg_graph(graph: nx.Graph) -> None:
    labels = {
        graph.nodes[n]["node_index"]: graph.nodes[n]["node_name"] for n in graph.nodes
    }
    node_types = [graph.nodes[n]["node_type"] for n in graph.nodes]

    # Create a list of colors based on the unique node types
    unique_node_types = list(set(node_types))
    color_map = {
        node_type: i / len(unique_node_types)
        for i, node_type in enumerate(unique_node_types)
    }
    colors = [colormaps["viridis"](color_map[node_type]) for node_type in node_types]

    fig = plt.figure()
    ax = fig.gca()

    nx.draw(graph, labels=labels, with_labels=True, node_color=colors, ax=ax)

    # Create a legend for the node colormap
    handles = []
    for node_type in unique_node_types:
        handle = patches.Patch(
            color=colormaps["viridis"](color_map[node_type]), label=node_type
        )
        handles.append(handle)
    plt.legend(handles=handles)

    plt.show()


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
    import pdb

    pdb.set_trace()
