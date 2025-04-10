import yaml
import argparse
from pathlib import Path
from typing import Any
from typing_extensions import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, AfterValidator

from bin import ROOT_DIR, GENERATION_CONFIG_PATHS, PROMPT_CONFIG_MAP
from semantic_kg import models
from semantic_kg.quality_control import utils
from semantic_kg.utils import load_subgraph_dataset
from semantic_kg.models.base import BaseTextGeneration
from semantic_kg.generation import (
    NLGenerationPipeline,
    default_exception_selector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        help=f"Name of dataset to load. Options are {list(GENERATION_CONFIG_PATHS.keys())}",
        required=False,
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        help="Optionally provide the path to a config directly",
        required=False,
    )
    parser.add_argument(
        "--n_responses",
        type=int,
        help="Number of responses to generate per subgraph",
        default=5,
    )
    parser.add_argument(
        "--max_quality_retries",
        type=int,
        default=5,
        help="Maximum number of retries if responses don't pass quality control",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        help="Path to save final dataset",
        default=ROOT_DIR / "datasets",
    )

    return parser.parse_args()


class LLMConfig(BaseModel):
    model_name: str
    params: dict[str, Any]

    def load(self, **kwargs) -> BaseTextGeneration:
        kwargs.update(self.params)
        return models.load_model(model_name=self.model_name, **kwargs)


class ScorerModelConfig(BaseModel):
    entity_extractor_model: LLMConfig
    kg_extractor_model: LLMConfig


def _dataset_name_validator(name: str) -> str:
    if name not in PROMPT_CONFIG_MAP:
        raise ValueError(
            f"Invalid dataset {name}. "
            f"Valid datasets are {list(PROMPT_CONFIG_MAP.keys())}"
        )

    return name


def _file_exists_validator(fpath: Path | str) -> Path:
    fpath = Path(fpath)
    if not fpath.exists():
        raise FileNotFoundError(f"{fpath} not found")

    return fpath


class DatasetConfig(BaseModel):
    dataset_name: Annotated[str, AfterValidator(_dataset_name_validator)]
    dataset_path: Annotated[Path | str, AfterValidator(_file_exists_validator)]
    response_model_config: LLMConfig
    scorer_model_config: ScorerModelConfig


def main(
    dataset_config: DatasetConfig,
    save_path: Path,
    n_responses: int = 10,
    max_quality_retries: int = 5,
) -> None:
    """Entry-point for generating final semantic-kg dataset from subgraph dataset"""
    prompt_config = PROMPT_CONFIG_MAP[dataset_config.dataset_name]

    subgraph_dataset = load_subgraph_dataset(dataset_config.dataset_path)

    response_model = dataset_config.response_model_config.load()
    entity_extractor_model = (
        dataset_config.scorer_model_config.entity_extractor_model.load(
            system_prompt=prompt_config.entity_extractor_scorer_prompt
        )
    )
    kg_extractor_model = dataset_config.scorer_model_config.kg_extractor_model.load(
        system_prompt=prompt_config.kg_extractor_scorer_prompt
    )

    # Quality control classes
    quality_scorers = utils.build_default_qc_scorers(
        entity_extractor_model=entity_extractor_model,
        kg_extractor_model=kg_extractor_model,
    )

    pipeline = NLGenerationPipeline(
        response_model,
        prompt_config.response_model_prompt,
        n_responses,
        backoff_exceptions=default_exception_selector(
            dataset_config.response_model_config.model_name  # type: ignore
        ),
        quality_scorers=quality_scorers,
        max_quality_retries=max_quality_retries,
    )
    df = pipeline.generate(subgraph_dataset)
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    load_dotenv()

    args = parse_args()
    if not args.dataset_name and not args.config_path:
        raise ValueError("`dataset_name` and `config_path` can't both be None")

    if args.dataset_name:
        config_path = GENERATION_CONFIG_PATHS[args.dataset_name]
    else:
        config_path = args.config_path

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset_config = DatasetConfig.model_validate(config)

    save_path = Path(args.save_path)
    if dataset_config.dataset_name not in str(save_path):
        save_path = (
            save_path
            / dataset_config.dataset_name
            / f"{dataset_config.dataset_name}_dataset.csv"
        )

    main(
        dataset_config=dataset_config,
        save_path=save_path,
        n_responses=args.n_responses,
        max_quality_retries=args.max_quality_retries,
    )
