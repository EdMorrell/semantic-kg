from pathlib import Path
from typing import Optional
from typing_extensions import Annotated

from pydantic import BaseModel, AfterValidator

from semantic_kg import prompts
from semantic_kg.datasets import EDGE_MAPPING_TYPE
from semantic_kg.datasets.oregano import OreganoLoader
from semantic_kg.datasets.prime_kg import PrimeKGLoader

ROOT_DIR = Path(__file__).parent.parent

DATASET_CONFIG_PATHS = {
    "oregano": ROOT_DIR / "config" / "datasets" / "oregano.yaml",
    "prime_kg": ROOT_DIR / "config" / "datasets" / "prime_kg.yaml",
}

GENERATION_CONFIG_PATHS = {
    "oregano": ROOT_DIR / "config" / "generation" / "oregano.yaml",
    "prime_kg": ROOT_DIR / "config" / "generation" / "prime_kg.yaml",
}

DATASET_LOADERS = {
    "oregano": OreganoLoader,
    "prime_kg": PrimeKGLoader,
}


PROMPT_CONFIG_MAP = {
    "oregano": prompts.OREGANO_PROMPT_CONFIG,
    "prime_kg": prompts.PRIME_KG_PROMPT_CONFIG,
}


def _dataset_name_validator(name: str) -> str:
    if name not in DATASET_LOADERS:
        raise ValueError(
            f"Invalid dataset {name}. Valid datasets are {list(DATASET_LOADERS.keys())}"
        )

    return name


def _dir_exists_validator(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    if not data_dir:
        raise NotADirectoryError(f"{data_dir} does not exist")

    return data_dir


class SubgraphDatasetConfig(BaseModel):
    dataset_name: Annotated[str, AfterValidator(_dataset_name_validator)]
    data_dir: Annotated[str | Path, AfterValidator(_dir_exists_validator)]
    src_node_id_field: str
    src_node_type_field: str
    src_node_name_field: str
    edge_name_field: str
    target_node_id_field: str
    target_node_type_field: str
    target_node_name_field: str
    directed_graph: bool

    edge_map: Optional[EDGE_MAPPING_TYPE] = None
    replace_map: Optional[EDGE_MAPPING_TYPE] = None
    valid_node_pairs: Optional[list[tuple[str, str]]] = None
