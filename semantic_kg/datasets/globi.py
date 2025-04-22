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

    def _replace_duplicate_ids(
        self, df: pd.DataFrame, id_field: str, name_field: str, type_field: str
    ) -> pd.DataFrame:
        """Finds any IDs that map to the same name and type, and replaces them with the
        first ID instance.

        # TODO: Make this a standalone function in utils
        """
        grp_df = df.groupby([name_field, type_field])[id_field].apply(
            lambda x: list(set(x))
        )
        ids = grp_df.to_list()
        replace_map = {}
        for id in ids:
            if len(id) == 1:
                replace_map[id[0]] = id[0]
            else:
                first = id[0]
                for dupl_id in id:
                    replace_map[dupl_id] = first

        df[id_field] = df[id_field].apply(lambda x: replace_map[x])
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

        # Remove entities with a missing ID ("no:match")
        df = df[
            (df["sourceTaxonId"] != "no:match") & (df["targetTaxonId"] != "no:match")
        ]

        # Merges any IDs that map to the same name and type
        df = self._replace_duplicate_ids(
            df, "sourceTaxonId", "sourceTaxonName", src_type_field
        )
        df = self._replace_duplicate_ids(
            df, "targetTaxonId", "targetTaxonName", target_type_field
        )

        return df


if __name__ == "__main__":
    loader = GlobiLoader("datasets/globi")
    df = loader.load()
