import sys
import warnings
from pathlib import Path
from typing import Optional
from typing_extensions import Annotated

from pydantic import BaseModel, AfterValidator

from semantic_kg import prompts

try:
    from semantic_kg.prompts import codex
except FileNotFoundError:
    warnings.warn(
        "Codex data currently missing. "
        "Please run `make download_dataset DATASET_NAME=codex` to use this dataset."
    )
try:
    from semantic_kg.prompts import codex
except FileNotFoundError:
    warnings.warn(
        "Globi data currently missing. "
        "Please run `make download_dataset DATASET_NAME=globi` to use this dataset."
    )
from semantic_kg.datasets import (
    EDGE_MAPPING_TYPE,
    OreganoLoader,
    PrimeKGLoader,
    CodexLoader,
    FindKGLoader,
    GlobiLoader,
)

ROOT_DIR = Path(__file__).parent.parent

DATASET_CONFIG_PATHS = {
    "oregano": ROOT_DIR / "config" / "datasets" / "oregano.yaml",
    "prime_kg": ROOT_DIR / "config" / "datasets" / "prime_kg.yaml",
    "codex": ROOT_DIR / "config" / "datasets" / "codex.yaml",
    "findkg": ROOT_DIR / "config" / "datasets" / "findkg.yaml",
    "globi": ROOT_DIR / "config" / "datasets" / "globi.yaml",
}

GENERATION_CONFIG_PATHS = {
    "oregano": ROOT_DIR / "config" / "generation" / "oregano.yaml",
    "prime_kg": ROOT_DIR / "config" / "generation" / "prime_kg.yaml",
    "codex": ROOT_DIR / "config" / "generation" / "codex.yaml",
    "findkg": ROOT_DIR / "config" / "generation" / "findkg.yaml",
    "globi": ROOT_DIR / "config" / "generation" / "globi.yaml",
}

DATASET_LOADERS = {
    "oregano": OreganoLoader,
    "prime_kg": PrimeKGLoader,
    "codex": CodexLoader,
    "findkg": FindKGLoader,
    "globi": GlobiLoader,
}


PROMPT_CONFIG_MAP = {
    "oregano": prompts.OREGANO_PROMPT_CONFIG,
    "prime_kg": prompts.PRIME_KG_PROMPT_CONFIG,
    "findkg": prompts.FINDKG_PROMPT_CONFIG,
}

# Codex and Globi are imported under a separate namespace to avoid slow imports
if "semantic_kg.prompts.codex" in sys.modules:
    PROMPT_CONFIG_MAP["codex"] = codex.CODEX_PROMPT_CONFIG
if "semantic_kg.prompts.globi" in sys.modules:
    PROMPT_CONFIG_MAP["globi"] = globi.GLOBI_PROMPT_CONFIG


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
    replace_map: Optional[dict[str, list[str]]] = None
    valid_node_pairs: Optional[list[tuple[str, str]]] = None
