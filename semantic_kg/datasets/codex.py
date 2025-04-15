from collections import Counter
import json
import itertools
from typing import Literal
from pathlib import Path
from functools import cached_property

import numpy as np
import pandas as pd

from semantic_kg.datasets.base import BaseDatasetLoader


class CodexLoader(BaseDatasetLoader):
    def __init__(
        self, data_dir: Path | str, size: Literal["s", "m", "l"] = "l"
    ) -> None:
        """Data loader for Codex dataset

        Parameters
        ----------
        data_dir : Path | str
            Path to directory containing codex data
        size : Literal["s", "m", "l"], optional
            Which sized dataset to load, by default "l"
        """
        super().__init__(data_dir)
        self.size = size

    @cached_property
    def entities(self) -> dict[str, dict[str, str]]:
        return json.load(open(self.data_dir / "entities" / "en" / "entities.json"))

    @cached_property
    def relations(self) -> dict[str, dict[str, str]]:
        return json.load(open(self.data_dir / "relations" / "en" / "relations.json"))

    @cached_property
    def entity_types(self) -> dict[str, str]:
        return json.load(open(self.data_dir / "types" / "entity2types.json"))

    @cached_property
    def entity_type_labels(self) -> dict[str, dict[str, str]]:
        return json.load(open(self.data_dir / "types" / "en" / "types.json"))

    def _load_triples(self, split: Literal["train", "valid", "test"]) -> pd.DataFrame:
        return pd.read_csv(
            self.data_dir / "triples" / f"codex-{self.size}" / f"{split}.txt",
            sep="\t",
            names=["head", "relation", "tail"],
            encoding="utf-8",
        )

    def _entity_label(self, eid: str) -> str:
        return self.entities[eid]["label"]

    def _relation_label(self, rid: str) -> str:
        return self.relations[rid]["label"]

    def _entity_type_label(self, type_id: str) -> str:
        return self.entity_type_labels[type_id]["label"]

    def _get_most_common_type(
        self, triple_df: pd.DataFrame, type_field: str
    ) -> pd.DataFrame:
        triple_types = list(itertools.chain.from_iterable(triple_df[type_field]))
        type_counts = Counter(triple_types)

        triple_df[type_field] = triple_df[type_field].apply(
            lambda x: x[np.argmax([type_counts[t] for t in x])]
        )

        return triple_df

    def _replace_duplicates(
        self, triple_df: pd.DataFrame, grp_df: pd.Series, field: str
    ) -> pd.DataFrame:
        """Replaces any duplicate IDs with the first ID instance"""
        ids = grp_df.to_list()
        replace_map = {}
        for id in ids:
            if len(id) == 1:
                replace_map[id[0]] = id[0]
            else:
                first = id[0]
                for dupl_id in id:
                    replace_map[dupl_id] = first

        triple_df[field] = triple_df[field].apply(lambda x: replace_map[x])
        return triple_df

    def _remove_duplicates(
        self, triple_df: pd.DataFrame, grp_df: pd.Series, field: str
    ) -> pd.DataFrame:
        """Removes all duplicates IDs except the first"""
        dupl_ids = grp_df[grp_df.apply(len) > 1].to_list()
        dupl_ids = list(itertools.chain.from_iterable(dupl_ids))
        return triple_df[~triple_df[field].isin(dupl_ids)]

    def _handle_duplicate_entries(
        self,
        triple_df: pd.DataFrame,
        field: Literal["head", "tail"],
        strategy: Literal["replace", "remove"],
    ) -> pd.DataFrame:
        """Handles head or tail IDs that map to the same name and type"""
        # Group IDs by name and type to find those which map to the same item
        grp_df = triple_df.groupby([f"{field}_name", f"{field}_type"])[field].apply(
            lambda x: list(set(x))
        )
        if strategy == "replace":
            return self._replace_duplicates(triple_df, grp_df, field)
        elif strategy == "remove":
            return self._remove_duplicates(triple_df, grp_df, field)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def load(self, most_common_entity_type: bool = True) -> pd.DataFrame:
        """Loads the Codex dataset as a dataframe of triples

        Parameters
        ----------
        most_common_entity_type : bool, optional
            If True then will only include the most frequent entity-type name
            for any entities with multiple types, by default True

        Returns
        -------
        pd.DataFrame
            Triple dataframe
        """
        triple_df = pd.concat(
            (
                self._load_triples(split="train"),
                self._load_triples(split="valid"),
                self._load_triples(split="test"),
            )
        )
        triple_df["head_name"] = triple_df["head"].apply(
            lambda x: self._entity_label(x)
        )
        triple_df["relation_name"] = triple_df["relation"].apply(
            lambda x: self._relation_label(x)
        )
        triple_df["tail_name"] = triple_df["tail"].apply(
            lambda x: self._entity_label(x)
        )

        triple_df["head_type"] = triple_df["head"].apply(
            lambda x: [self._entity_type_label(t) for t in self.entity_types[x]]
        )

        triple_df["tail_type"] = triple_df["tail"].apply(
            lambda x: [self._entity_type_label(t) for t in self.entity_types[x]]
        )

        if most_common_entity_type:
            triple_df = self._get_most_common_type(triple_df, "head_type")
            triple_df = self._get_most_common_type(triple_df, "tail_type")

        triple_df = self._handle_duplicate_entries(triple_df, "head", "replace")
        triple_df = self._handle_duplicate_entries(triple_df, "tail", "replace")

        return triple_df


if __name__ == "__main__":
    loader = CodexLoader("datasets/codex")
    df = loader.load()
