import random
from typing import Optional

import pandas as pd
import networkx as nx
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import colormaps

from semantic_kg.llm import OpenAITextGeneration
from semantic_kg.prompts import triplet_prompt_template
from semantic_kg.sampling import SubgraphSampler, generate_triplets
from semantic_kg.perturbation import (
    EdgeAdditionPerturbation,
    EdgeDeletionPerturbation,
    EdgeReplacementPerturbation,
    NodeRemovalPerturbation,
    GraphPerturber,
)


NODE_DPATH = "~/Downloads/dataverse_files/nodes.csv"
EDGE_DPATH = "~/Downloads/dataverse_files/edges.csv"


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
    """Creates a mapping from "relation" to "display_relation""""
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
        node_name_id="node_type",
        valid_edges=list(relation_map.keys()),
        edge_attribute_mapper=PrimeKGEdgeAttributeMapper(
            relation_map=relation_map, display_relation_map=display_relation_map
        ),
    )
    edge_deletion_perturbation = EdgeDeletionPerturbation()
    edge_replacement_perturbation = EdgeReplacementPerturbation(
        node_name_id="node_type",
        edge_name_id="relation",
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
        p_prob=[0.3, 0.3, 0.3, 0.1]
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


if __name__ == "__main__":
    load_dotenv()

    node_df = pd.read_csv(NODE_DPATH)
    edge_df = pd.read_csv(EDGE_DPATH)

    relation_map = create_relation_map(node_df, edge_df)
    display_relation_map = create_display_relation_map(edge_df)

    # Any relations with multiple allowed values added to `replace_map`
    replace_map = {k: v for k, v in relation_map.items() if len(v) > 1}

    prime_kg_loader = PrimeKGLoader(node_df=node_df, edge_df=edge_df)
    prime_kg = prime_kg_loader.load()

    # Only select disease nodes as start nodes
    start_node = random.choice(node_df[node_df["node_type"] == "disease"].index)

    subgraph_sampler = SubgraphSampler(
        prime_kg, node_name_field="node_name", edge_name_field="display_relation"
    )

    subgraph = subgraph_sampler.sample(
        n_nodes=10, start_node=start_node, max_neighbours=3
    )

    perturber = build_primekg_perturber(relation_map, display_relation_map, replace_map)
    perturbed_subgraph = perturber.perturb(subgraph, n_perturbations=5)

    plot_primekg_graph(subgraph)
    plot_primekg_graph(perturbed_subgraph)

    triplets = generate_triplets(
        subgraph,
        "node_name",
        "display_relation",
        node_attr_fields=["node_type"],
        edge_attr_fields=["relation"],
    )
    perturbed_triplets = generate_triplets(
        perturbed_subgraph,
        "node_name",
        "display_relation",
        node_attr_fields=["node_type"],
        edge_attr_fields=["relation"],
    )

    llm = OpenAITextGeneration(model_id="gpt-4-32k")
    prime_kg_prompt_rules = """If the relation is described as "parent-child" this refers to the fact that the "source_node" is a sub-type of the "target_node", it does not refer to hereditary relations.

    Please ensure all nodes are listed in natural-language and do not use quotes to describe entities.
    """

    prompt = triplet_prompt_template.format(
        dataset_rules=prime_kg_prompt_rules, triplets=triplets
    )

    response = llm.generate(prompt, n_responses=1, max_tokens=500)

    perturbed_prompt = triplet_prompt_template.format(
        dataset_rules=prime_kg_prompt_rules, triplets=perturbed_triplets
    )
    perturbed_response = llm.generate(perturbed_prompt, n_responses=1, max_tokens=500)

    print(response[0])
    print(perturbed_response[0])
