from pathlib import Path

import pandas as pd

from semantic_kg.datasets import BaseDatasetLoader


class FindKGDatasetLoader(BaseDatasetLoader):
    def __init__(self, data_dir: Path | str) -> None:
        """Dataset loader for Financial Dynamic Knowledge-Graph

        See https://xiaohui-victor-li.github.io/FinDKG for more info

        Parameters
        ----------
        data_dir : Path | str
            Directory containing data

        Raises
        ------
        ValueError
            If any expected file in data-directory does not exist
        """
        super().__init__(data_dir)
        self.triple_path = self.data_dir / "train.txt"
        if not self.triple_path.exists():
            raise ValueError(f"{self.triple_path} does not exist")

        self.entity_map_path = self.data_dir / "entity2id.txt"
        if not self.entity_map_path.exists():
            raise ValueError(f"{self.entity_map_path} does not exist")

        self.relation_map_path = self.data_dir / "relation2id.txt"
        if not self.relation_map_path.exists():
            raise ValueError(f"{self.relation_map_path} does not exist")

    def _create_id_map(
        self, map_df: pd.DataFrame, id_col: str, name_col: str
    ) -> dict[int, str]:
        return map_df[[id_col, name_col]].set_index(id_col).to_dict()[name_col]

    def load(self) -> pd.DataFrame:
        """Load the FindKG dataset."""
        triple_df = pd.read_csv(
            self.triple_path,
            sep="\t",
            header=None,
            names=["subject_id", "relation_id", "object_id", "time", "None"],
            index_col="subject_id",
        ).reset_index()
        entity_df = pd.read_csv(
            self.entity_map_path,
            sep="\t",
            header=None,
            names=["Name", "ID", "Type", "None"],
        )
        relation_df = pd.read_csv(
            self.relation_map_path, sep="\t", header=None, names=["Name", "ID"]
        )

        entity_map = self._create_id_map(entity_df, id_col="ID", name_col="Name")
        entity_type_map = self._create_id_map(entity_df, id_col="ID", name_col="Type")
        relation_map = self._create_id_map(relation_df, id_col="ID", name_col="Name")

        triple_df["subject_name"] = triple_df["subject_id"].apply(
            lambda x: entity_map[x]
        )
        triple_df["object_name"] = triple_df["object_id"].apply(lambda x: entity_map[x])
        triple_df["subject_type"] = triple_df["subject_id"].apply(
            lambda x: entity_type_map[x]
        )
        triple_df["object_type"] = triple_df["object_id"].apply(
            lambda x: entity_type_map[x]
        )

        triple_df["relation"] = triple_df["relation_id"].apply(
            lambda x: relation_map[x]
        )

        return triple_df[
            [
                "subject_id",
                "subject_name",
                "subject_type",
                "relation",
                "object_id",
                "object_name",
                "object_type",
            ]
        ]


if __name__ == "__main__":
    loader = FindKGDatasetLoader("datasets/findkg")
    triple_df = loader.load()
