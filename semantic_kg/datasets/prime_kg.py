from pathlib import Path
import random

import pandas as pd

from semantic_kg.datasets import BaseDatasetLoader


class PrimeKGLoader(BaseDatasetLoader):
    def __init__(self, data_dir: str | Path) -> None:
        super().__init__(data_dir=data_dir)
        self.node_fpath = self.data_dir / "nodes.csv"
        if not self.node_fpath.exists():
            raise ValueError(f"{self.node_fpath} does not exist")
        self.edge_fpath = self.data_dir / "edges.csv"
        if not self.edge_fpath.exists():
            raise ValueError(f"{self.edge_fpath} does not exist")

    def load(self) -> pd.DataFrame:
        node_df = pd.read_csv(self.node_fpath, sep="\t")
        edge_df = pd.read_csv(self.edge_fpath)

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

    from semantic_kg.generation import SubgraphPipeline

    random.seed(42)
    np.random.seed(42)

    primekg_loader = PrimeKGLoader("datasets/prime_kg")
    df = primekg_loader.load()

    pipeline = SubgraphPipeline(
        src_node_id_field="src_node_index",
        src_node_type_field="src_node_type",
        src_node_name_field="src_node_name",
        edge_name_field="display_relation",
        target_node_id_field="target_node_index",
        target_node_type_field="target_node_type",
        target_node_name_field="target_node_name",
        directed=False,
    )
    subgraph_df = pipeline.generate(df, 1000)
