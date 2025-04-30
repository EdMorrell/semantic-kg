from typing import Optional

import pandas as pd
import networkx as nx

from semantic_kg.datasets.prime_kg import PrimeKGLoader  # noqa: F401
from semantic_kg.datasets.oregano import OreganoLoader  # noqa: F401
from semantic_kg.datasets.codex import CodexLoader  # noqa: F401
from semantic_kg.datasets.findkg import FindKGLoader  # noqa: F401
from semantic_kg.datasets.globi import GlobiLoader  # noqa: F401


EDGE_MAPPING_TYPE = dict[tuple[str, str], list[str]]


class KGLoader:
    def __init__(
        self,
        src_node_id_field: str,
        src_node_type_field: str,
        src_node_name_field: Optional[str],
        edge_name_field: str,
        target_node_id_field: str,
        target_node_type_field: str,
        target_node_name_field: Optional[str],
    ):
        self.src_node_id_field = src_node_id_field
        self.src_node_type_field = src_node_type_field
        self.src_node_name_field = src_node_name_field
        self.edge_name_field = edge_name_field
        self.target_node_id_field = target_node_id_field
        self.target_node_type_field = target_node_type_field
        self.target_node_name_field = target_node_name_field

    def _validate_fields(self, triple_df: pd.DataFrame) -> None:
        if self.src_node_id_field not in triple_df.columns:
            raise ValueError(
                f"No field named {self.src_node_id_field} found in dataframe"
            )
        if self.src_node_type_field not in triple_df.columns:
            raise ValueError(
                f"No field named {self.src_node_type_field} found in dataframe"
            )
        if (
            self.src_node_name_field
            and self.src_node_name_field not in triple_df.columns
        ):
            raise ValueError(
                f"No field named {self.src_node_name_field} found in dataframe"
            )

        if self.target_node_id_field not in triple_df.columns:
            raise ValueError(
                f"No field named {self.target_node_id_field} found in dataframe"
            )

    def _create_node_attribute_map(
        self, triple_df: pd.DataFrame
    ) -> dict[str, dict[str, str]]:
        src_node_attrs = [self.src_node_id_field, self.src_node_type_field]
        if self.src_node_name_field:
            src_node_attrs.append(self.src_node_name_field)

        node_attribute_map = (
            triple_df[src_node_attrs]
            .drop_duplicates()
            .set_index(self.src_node_id_field)
            .rename(
                # Rename to ensure all name and type fields the same
                {
                    self.src_node_type_field: "node_type",
                    self.src_node_name_field: "node_name",
                },
                axis=1,
            )
            .to_dict(orient="index")
        )

        target_node_attrs = [self.target_node_id_field, self.target_node_type_field]
        if self.target_node_name_field:
            target_node_attrs.append(self.target_node_name_field)

        target_node_attribute_map = (
            triple_df[target_node_attrs]
            .drop_duplicates()
            .set_index(self.target_node_id_field)
            .rename(
                {
                    self.target_node_type_field: "node_type",
                    self.target_node_name_field: "node_name",
                },
                axis=1,
            )
            .drop_duplicates()
            .to_dict(orient="index")
        )

        node_attribute_map.update(target_node_attribute_map)

        return node_attribute_map  # type: ignore

    def load(self, triple_df: pd.DataFrame, directed: bool = True) -> nx.Graph:
        self._validate_fields(triple_df)

        triple_df = triple_df.rename({self.edge_name_field: "edge_name"}, axis=1)

        kwargs = {}
        if directed:
            kwargs["create_using"] = nx.DiGraph

        g = nx.from_pandas_edgelist(
            df=triple_df,
            source=self.src_node_id_field,
            target=self.target_node_id_field,
            edge_attr=["edge_name"],
            **kwargs,
        )

        node_attr_map = self._create_node_attribute_map(triple_df)

        nx.set_node_attributes(g, node_attr_map)

        return g


def get_valid_node_pairs(
    triple_df: pd.DataFrame, src_node_type_field: str, target_node_type_field: str
) -> list[tuple[str, str]]:
    return list(
        set(
            triple_df[[src_node_type_field, target_node_type_field]].itertuples(
                index=False, name=None
            )
        )
    )
