import json
import random
import requests
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm
from hishel import CacheClient

from semantic_kg.perturbation import (
    build_perturber,
)
from semantic_kg.utils import get_hishel_http_client


OREGANO_REPLACE_MAP: dict[tuple[str, str], list[str]] = {
    ("COMPOUND", "ACTIVITY"): ["increase_activity", "decrease_activity"],
    ("COMPOUND", "COMPOUND"): ["increase_activity", "decrease_activity"],
    ("COMPOUND", "EFFECT"): ["increase_effect", "decrease_effect"],
}


def load_triple_df(fpath: str) -> pd.DataFrame:
    df = pd.read_csv(fpath, sep="\t", header=None)
    df = df.rename(columns={0: "subject", 1: "predicate", 2: "object"})
    df = df[df["predicate"] != "has_code"]
    return df


def get_reactome_pathway_name(entity: str, client: Optional[CacheClient] = None) -> str:
    """Makes a request to REACTOME to get the name of a Pathway entity

    Parameters
    ----------
    entity : str
        Entity to get name of (e.g. "R-HSA-390650")
    client : Optional[CacheClient], optional
        Optionally provide an HTTP client for caching requests, by default None

    Returns
    -------
    str
        A string denoting the `displayName` of the entity

    Raises
    ------
    requests.exceptions.HTTPError:
        If request fails to return a valid response
    ValueError
        If response is empty
    KeyError
        If response missing a field called `displayName`
    """
    headers = {
        "accept": "*/*",
    }

    get_func = client.get if client else requests.get
    response = get_func(
        url=f"https://reactome.org/ContentService/data/pathways/low/entity/{entity}",
        headers=headers,
    )
    response.raise_for_status()

    content = json.loads(response.content.decode())
    if len(content) == 0:
        raise ValueError("Empty content")

    content = content[0]
    if "displayName" not in content:
        raise KeyError("Missing 'displayName' field")

    return content["displayName"]


def get_ncbi_gene_name(ncbi_id: str, client: Optional[CacheClient] = None) -> str:
    """Makes a request to NCBI to get the name of a gene entity

    Parameters
    ----------
    ncbi_id : str
        The NCBI ID of the gene
    client : Optional[CacheClient], optional
        Optionally provide an HTTP client for caching requests, by default None

    Returns
    -------
    str
        The name of the gene

    Raises
    ------
    requests.exceptions.HTTPError:
        If request fails to return a valid response
    ValueError
        If unable to parse the gene name from the response
    """
    get_func = client.get if client else requests.get

    response = get_func(
        url=f"https://www.ncbi.nlm.nih.gov/gene/?term={ncbi_id}&report=docsum&format=text",
    )
    response.raise_for_status()

    text = response.content.decode()
    try:
        # gene_name = re.split(r"Official Symbol: ([A-Z0-9a-z]+) and Name", text)[1]
        gene_name = text.split("<pre>")[1].strip().split("\n")[0].split("1.")[1].strip()
    except IndexError:
        raise ValueError(f"Unable to parse gene name from repsonse for {ncbi_id}")

    return gene_name


def _extract_entity_map_from_metadata(fpath: str) -> dict[str, str]:
    """Extracts a map from entity name to display name

    Parameters
    ----------
    fpath : str
        Path to metadata .ttl file (default version is called
        "oreganov2.1_metadata_complete.ttl")

    Returns
    -------
    dict[str, str]
        Mapping from entity name to display map (e.g. {"COMPOUND:1000": "minocycline"})
    """
    result = {}

    with open(fpath, "r") as file:
        file_content = []
        for line in file:
            # Skips the first lines which contain unnecessary metadata
            if line.startswith("@prefix"):
                continue
            else:
                file_content.append(line)
    file_content = "".join(file_content)

    # Splits the file content into double-line separated entity-blocks
    blocks = file_content.strip().split("\n\n")

    # Finds the relevant information in the header of each block
    for block in blocks:
        first_line = block.split(";")[0]

        components = first_line.split("\t")

        entity_type, entity_id_full = components[0].split(":")
        entity_id = entity_id_full.split("_")[-1]

        labels_part = components[2].strip().strip('";')
        labels = [label.strip('"') for label in labels_part.split(", ")]

        dict_key = f"{entity_type.upper()}:{entity_id}"

        preferred_name = labels[1] if len(labels) > 1 else labels[0]

        # Some entities do not include the display name so we skip them
        # This try/except block will catch any entities missing this
        try:
            preferred_name = preferred_name.split(":")[1]
        except IndexError:
            continue

        result[dict_key] = preferred_name

    return result


def _remove_id_strings(entity_map: dict[str, str]) -> dict[str, str]:
    """Removes InChIKey entities (this will create some FPs)"""
    return {k: v for k, v in entity_map.items() if len(v) != 27}


class OreganoLoader:
    NODE_ATTR_MAP = {
        "activity": {
            "fname": "ACTIVITY.tsv",
            "id_field_name": "ID_OREGANO:78",
            "attr_field_name": "NAME_DRUGBANK",
        },
        "effect": {
            "fname": "EFFECT.tsv",
            "id_field_name": "ID_OREGANO:171",
            "attr_field_name": "NAME_DRUGBANK",
        },
        "indication": {
            "fname": "INDICATION.tsv",
            "id_field_name": "ID_OREGANO:2714",
            "attr_field_name": "NAME",
        },
        "side_effect": {
            "fname": "SIDE_EFFECT.tsv",
            "id_field_name": "ID_OREGANO:6060",
            "attr_field_name": "NAME",
        },
    }

    def __init__(
        self,
        data_dir: Path | str,
        incl_pathway_data: bool = True,
        incl_ncbi_genes: bool = True,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"{data_dir} is not a directory")

        # Configures path to required files for dataset
        triple_fpath = self.data_dir / "OREGANO_V2.1.tsv"
        if not triple_fpath.is_file():
            raise ValueError(f"No triple file found in directory {data_dir}")
        self.triple_fpath = triple_fpath

        metadata_fpath = self.data_dir / "oreganov2.1_metadata_complete.ttl"
        if not metadata_fpath.is_file():
            raise ValueError(f"No metadata file found in directory {data_dir}")
        self.metadata_fpath = metadata_fpath

        # Configures path to optional files for dataset
        self._set_annotation_fpaths()

        # Sets up cache directory
        if not cache_dir:
            cache_path = Path("outputs/ncbi_responses")
        else:
            cache_path = Path(cache_dir)

        cache_path.mkdir(parents=True, exist_ok=True)
        self.cache_client = get_hishel_http_client(cache_dir=cache_path)

        # Configures path to files that require network calls to retrieve data
        self.incl_pathway_data = incl_pathway_data
        if self.incl_pathway_data:
            pathway_fpath = self.data_dir / "PATHWAYS.tsv"
            if not pathway_fpath.is_file():
                raise ValueError(f"No pathway file found in directory {data_dir}")
            self.pathway_fpath = pathway_fpath

        self.incl_ncbi_genes = incl_ncbi_genes
        if self.incl_ncbi_genes:
            gene_fpath = self.data_dir / "GENES.tsv"
            if not gene_fpath.is_file():
                raise ValueError(f"No genes file found in directory {data_dir}")
            self.gene_fpath = gene_fpath

    def _set_annotation_fpaths(self) -> None:
        """Checks whether annotation files exists and sets filenames if so"""
        for attr_val, meta in self.NODE_ATTR_MAP.items():
            fpath = self.data_dir / meta["fname"]
            if not fpath.is_file():
                print(
                    f"No file found for {meta['fname']}. Will not be able to set metadata."
                )
            else:
                cls_attr_name = f"{attr_val}_fpath"
                setattr(self, cls_attr_name, fpath)

    def _load_entity_map(self) -> dict[str, str]:
        # Loads a map of ID to metadata from metadata file
        entity_map = _extract_entity_map_from_metadata(str(self.metadata_fpath))

        # Adds any available data from attribute mapping files
        for attr_val, meta in self.NODE_ATTR_MAP.items():
            cls_attr_name = f"{attr_val}_fpath"
            if hasattr(self, cls_attr_name):
                fpath = getattr(self, cls_attr_name)
                df = pd.read_csv(fpath, sep="\t")

                id_field = meta["id_field_name"]
                name_field = meta["attr_field_name"]

                attr_map = (
                    df[[id_field, name_field]].set_index(id_field).to_dict()[name_field]
                )
                entity_map.update(attr_map)

        return entity_map

    def _get_pathway_map(self, df: pd.DataFrame) -> dict[str, str]:
        pathway_ids = (
            df[df["object"].apply(lambda x: x.split(":")[0] == "PATHWAY")]["object"]
            .unique()
            .tolist()
        )
        pathway_ids.extend(
            df[df["subject"].apply(lambda x: x.split(":")[0] == "PATHWAY")]["subject"]
            .unique()
            .tolist()
        )

        pathway_df = pd.read_csv(self.pathway_fpath, sep="\t")
        reactome_id_map = pathway_df.set_index("ID_OREGANO:2129").to_dict()["REACTOME"]

        print("Getting pathway names from REACTOME")
        pathway_map = {}
        for pathway_id in tqdm(pathway_ids):
            try:
                reactome_id = reactome_id_map[pathway_id]
                pathway_map[pathway_id] = get_reactome_pathway_name(
                    reactome_id,
                    self.cache_client,
                )
            except requests.exceptions.HTTPError:
                print(f"Unable to get pathway name for {pathway_id}")
                continue

        return pathway_map

    def _load_pathway_map(self, df: pd.DataFrame) -> dict[str, str]:
        pathway_map_fpath = self.data_dir / "pathway_map.json"
        if not pathway_map_fpath.is_file():
            pathway_map = self._get_pathway_map(df)
            with open(pathway_map_fpath, "w") as file:
                json.dump(pathway_map, file)
        else:
            with open(pathway_map_fpath, "r") as file:
                pathway_map = json.load(file)

        return pathway_map

    def _get_ncbi_gene_map(
        self, df: pd.DataFrame, entity_map: dict[str, str]
    ) -> dict[str, str]:
        # Finds gene names with missing metadata
        all_gene_entities = (
            df[df["subject"].apply(lambda x: "GENE" in x)]["subject"].unique().tolist()
        )
        all_gene_entities.extend(
            df[df["object"].apply(lambda x: "GENE" in x)]["object"].unique().tolist()
        )
        all_gene_entities = set(all_gene_entities)

        existing_gene_entities = {k for k in entity_map.keys() if "GENE" in k}

        missing_genes = all_gene_entities - existing_gene_entities

        gene_df = pd.read_csv(self.gene_fpath, sep="\t", dtype={"NCBI GENE": object})
        gene_df = gene_df.dropna(subset=["NCBI GENE"])

        ncbi_gene_map = gene_df.set_index("ID_OREGANO:35794").to_dict()["NCBI GENE"]

        print("Getting missing genes from NCBI")
        gene_map = {}
        for gene in tqdm(missing_genes):
            if gene in ncbi_gene_map:
                try:
                    gene_map[gene] = get_ncbi_gene_name(ncbi_gene_map[gene])
                except (requests.exceptions.HTTPError, ValueError):
                    print(f"Unable to get gene name for {gene}:{ncbi_gene_map[gene]}")
                    continue
            else:
                print(f"No NCBI data available for {gene}")
                continue

        return gene_map

    def _load_gene_map(
        self, df: pd.DataFrame, entity_map: dict[str, str]
    ) -> dict[str, str]:
        gene_map_fpath = self.data_dir / "gene_map.json"
        if not gene_map_fpath.is_file():
            gene_map = self._get_ncbi_gene_map(df, entity_map)
            with open(gene_map_fpath, "w") as file:
                json.dump(gene_map, file)
        else:
            with open(gene_map_fpath, "r") as file:
                gene_map = json.load(file)

        return gene_map

    def _merge_duplicate_entities(
        self, df: pd.DataFrame, entity_map: dict[str, str]
    ) -> pd.DataFrame:
        # Find all entities that map to the same value
        reverse_map = {}
        for k, v in entity_map.items():
            reverse_map.setdefault(v, []).append(k)

        # Finds instances of duplicates in dataframe and merges into first instance
        replace_map = {}
        for k, v in reverse_map.items():
            if len(v) < 2:
                continue

            # Map ensure we only merge duplicate entities if they're of the same type
            first_instance_type_map = {v[0].split(":")[0]: v[0]}
            for id in v[1:]:
                id_type = id.split(":")[0]
                if id_type in first_instance_type_map:
                    replace_map[id] = first_instance_type_map[id_type]
                else:
                    first_instance_type_map[id_type] = id

        df["subject"] = df["subject"].map(lambda x: replace_map.get(x, x))
        df["object"] = df["object"].map(lambda x: replace_map.get(x, x))

        return df

    def load(
        self, detect_ids: bool = True, max_entity_len: Optional[int] = 50
    ) -> pd.DataFrame:
        df = load_triple_df(str(self.triple_fpath))

        entity_map = self._load_entity_map()

        if self.incl_pathway_data:
            pathway_map = self._load_pathway_map(df)
            entity_map.update(pathway_map)

        # Only use nodes we have metadata for
        if self.incl_ncbi_genes:
            ncbi_gene_map = self._load_gene_map(df, entity_map)
            entity_map.update(ncbi_gene_map)

        if detect_ids:
            entity_map = _remove_id_strings(entity_map)

        if max_entity_len:
            entity_map = {
                k: v for k, v in entity_map.items() if len(v) < max_entity_len
            }

        entities = set(entity_map.keys())
        df = df[(df["subject"].isin(entities)) & (df["object"].isin(entities))]

        df = self._merge_duplicate_entities(df, entity_map)

        df["subject_name"] = df["subject"].map(entity_map)
        df["object_name"] = df["object"].map(entity_map)
        df["subject_type"] = df["subject"].apply(lambda x: x.split(":")[0])
        df["object_type"] = df["object"].apply(lambda x: x.split(":")[0])

        df = df.rename({"subject": "subject_id", "object": "object_id"}, axis=1)
        return df[
            [
                "subject_id",
                "subject_name",
                "subject_type",
                "predicate",
                "object_id",
                "object_name",
                "object_type",
            ]
        ]


if __name__ == "__main__":
    import numpy as np
    from semantic_kg.sampling import SubgraphDataset, SubgraphSampler
    from semantic_kg.datasets import KGLoader, create_edge_map, get_valid_node_pairs

    random.seed(42)
    np.random.seed(42)

    oregano_loader = OreganoLoader("datasets/oregano")
    df = oregano_loader.load()

    kg_loader = KGLoader(
        src_node_id_field="subject_id",
        src_node_type_field="subject_type",
        src_node_name_field="subject_name",
        edge_name_field="predicate",
        target_node_id_field="object_id",
        target_node_type_field="object_type",
        target_node_name_field="object_name",
    )
    g = kg_loader.load(triple_df=df, directed=True)

    edge_map = kg_loader.create_edge_map(df, directed=True)
    edge_map = create_edge_map(
        df,
        src_node_type_field="subject_type",
        target_node_type_field="object_type",
        edge_name_field="predicate",
    )
    replace_map = {k: v for k, v in edge_map.items() if len(v) > 1}
    valid_node_pairs = get_valid_node_pairs(
        df, src_node_type_field="subject_type", target_node_type_field="object_type"
    )

    perturber = build_perturber(
        edge_map=edge_map,
        valid_node_pairs=valid_node_pairs,  # type: ignore
        replace_map=OREGANO_REPLACE_MAP,
        directed=True,
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
