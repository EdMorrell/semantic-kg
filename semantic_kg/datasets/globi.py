from pathlib import Path
from typing import Literal

import pandas as pd

from semantic_kg.datasets.base import BaseDatasetLoader


class GlobiLoader(BaseDatasetLoader):
    def __init__(
        self,
        data_dir: Path | str,
        rank_level: Literal[
            "kingdom", "phylum", "class", "order", "family", "genus", "species"
        ] = "species",
        type_rank: Literal[
            "Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"
        ] = "Family",
    ) -> None:
        """Loader for the Global Biotics Interaction dataset

        See: https://www.globalbioticinteractions.org/

        Parameters
        ----------
        data_dir : Path | str
            Directory to Globi Data
        rank_level : Literal[
            "kingdom", "phylum", "class", "order", "family", "genus", "species"
        ]
            The taxonomical rank to load data for. For example if "species", will only
            load "species" enities. Defaults to species.
        type_rank : Literal[
            "Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"
        ]
            The taxonomical rank to consider as the node-type. This can be used for
            ensuring node-replacements stay within a taxonomical rank. Defaults to
            "Family".
        """
        super().__init__(data_dir)
        self.fpath = Path(data_dir) / "interactions.csv"
        if not self.fpath.exists():
            raise ValueError(f"{self.fpath} does not exist")
        self.rank_level = rank_level
        self.type_rank = type_rank

    def _load_from_full(
        self, src_type_field: str, target_type_field: str, rank_level_fpath: Path
    ) -> pd.DataFrame:
        load_cols = [
            "sourceTaxonId",
            src_type_field,
            "sourceTaxonName",
            "interactionTypeName",
            "targetTaxonId",
            target_type_field,
            "targetTaxonName",
        ]
        df = pd.read_csv(
            self.fpath, usecols=load_cols + ["sourceTaxonRank", "targetTaxonRank"]
        )
        # Only use rows where entities are at `self.rank_level` level
        df = df[
            (df["sourceTaxonRank"] == self.rank_level)
            & (df["targetTaxonRank"] == self.rank_level)
        ][load_cols]

        # Cache for quicker future loading
        df.to_csv(rank_level_fpath, index=False)

        return df

    def load(self) -> pd.DataFrame:
        src_type_field = f"sourceTaxon{self.type_rank}Name"
        target_type_field = f"targetTaxon{self.type_rank}Name"
        # Checks whether a cached rank-level CSV exists
        rank_level_fpath = self.data_dir / f"interactions_{self.rank_level}.csv"
        if rank_level_fpath.exists():
            df = pd.read_csv(rank_level_fpath)
        else:
            df = self._load_from_full(
                src_type_field, target_type_field, rank_level_fpath
            )

        # Remove rows with null type-fields
        df = df.dropna(axis=0, subset=[src_type_field, target_type_field])

        return df


if __name__ == "__main__":
    from semantic_kg.datasets import create_edge_map

    loader = GlobiLoader("datasets/globi")
    df = loader.load()

    edge_map = create_edge_map(
        df,
        src_node_type_field="sourceTaxonFamilyName",
        target_node_type_field="targetTaxonFamilyName",
        edge_name_field="interactionTypeName",
        directed=True,
    )
    replace_map = {k: v for k, v in edge_map.items() if len(v) > 1}
