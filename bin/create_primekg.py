import argparse
import ast
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from semantic_kg import models
from semantic_kg.quality_control import (
    RegexMatcherScorer,
    QUOTE_REGEX,
    build_reconstruction_scorer,
)
from semantic_kg.models import OpenAITextGeneration
from semantic_kg.prompts import prime_kg
from semantic_kg.prompts.default import triple_response_format
from semantic_kg.datasets.prime_kg import (
    format_primekg_prompt,
)
from semantic_kg.generation import (
    NLGenerationPipeline,
    default_exception_selector,
    ScorerConfig,
)


ROOT_DIR = Path(__file__).parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subgraph_fpath",
        type=Path,
        help="Path to subgraph dataset",
        default=ROOT_DIR
        / "datasets"
        / "prime_kg"
        / "73fef7a77fd16f36852cbfad309f976d.csv",
    )
    parser.add_argument(
        "--node_fpath",
        type=Path,
        help="Path to PrimeKG nodes CSV file for finding node types for scorer",
        default=ROOT_DIR / "datasets" / "prime_kg" / "nodes.csv",
    )
    parser.add_argument(
        "--edge_fpath",
        type=Path,
        help="Path to PrimeKG edges CSV file for finding edge types for scorer",
        default=ROOT_DIR / "datasets" / "prime_kg" / "edges.csv",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help=f"Model type to load. Options are: {list(models.MODEL_MAP.keys())}",
        default="openai",
    )
    parser.add_argument(
        "--model_id", type=str, help="ID of model to load", default="gpt-4-32k"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to cache model responses",
        default=ROOT_DIR / "outputs" / "responses",
    )
    parser.add_argument(
        "--n_responses",
        type=int,
        help="Number of responses to generate per subgraph",
        default=10,
    )
    parser.add_argument(
        "--max_quality_retries",
        type=int,
        default=5,
        help="Maximum number of retries if responses don't pass quality control",
    )
    parser.add_argument(
        "--match_direction",
        type=bool,
        help="Whether to match directed graphs during quality-control, otherwise checks for match ignoring direction",
        default=True,
    )
    parser.add_argument(
        "--temperature", type=float, help="Temperature to use during generation"
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        help="Path to save final dataset",
        default=ROOT_DIR / "datasets" / "prime_kg" / "prime_kg_dataset.csv",
    )

    return parser.parse_args()


def build_qc_scorers(
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    match_direction: bool,
) -> list[ScorerConfig]:
    """Build quality control scorers"""
    node_types = node_df["node_type"].unique().tolist()
    edge_types = edge_df["display_relation"].unique().tolist()

    system_prompt = prime_kg.prime_kg_scorer_system_prompt_template.format(
        node_types=node_types,
        edge_types=edge_types,
        response_schema=triple_response_format,
    )

    scorer_model = OpenAITextGeneration(
        model_id="gpt-4o",
        cache_dir=Path("outputs/responses/"),
        temperature=1.0,
        system_prompt=system_prompt,
        structured_output=True,
    )
    reconstruction_scorer = build_reconstruction_scorer(scorer_model, match_direction)

    regex_scorer = RegexMatcherScorer(pattern=QUOTE_REGEX, mode="not_exists")

    return [
        ScorerConfig(scorer=reconstruction_scorer, accept_threshold=1.0),
        ScorerConfig(scorer=regex_scorer, accept_threshold=1.0),
    ]


def main(
    subgraph_fpath: Path,
    node_fpath: Path,
    edge_fpath: Path,
    model_type: str,
    model_id: str,
    cache_dir: Path,
    save_path: Path,
    n_responses: int = 10,
    max_quality_retries: int = 5,
    match_direction: bool = True,
    temperature: float = 1.0,
) -> None:
    """Script to generate responses from an LLM to PrimeKG subgraphs"""
    subgraph_dataset = pd.read_csv(subgraph_fpath)

    # TODO: Create a load function to handle this
    subgraph_dataset["subgraph_triples"] = subgraph_dataset["subgraph_triples"].apply(
        ast.literal_eval
    )
    subgraph_dataset["perturbed_subgraph_triples"] = subgraph_dataset[
        "perturbed_subgraph_triples"
    ].apply(ast.literal_eval)

    node_df = pd.read_csv(node_fpath)
    edge_df = pd.read_csv(edge_fpath)

    model = models.load_model(
        model_type, model_id=model_id, cache_dir=cache_dir, temperature=temperature
    )
    prompt_template = format_primekg_prompt()

    # Quality control classes
    quality_scorers = build_qc_scorers(node_df, edge_df, match_direction)

    pipeline = NLGenerationPipeline(
        model,
        prompt_template,
        n_responses,
        backoff_exceptions=default_exception_selector(model_type),  # type: ignore
        quality_scorers=quality_scorers,
        max_quality_retries=max_quality_retries,
    )
    df = pipeline.generate(subgraph_dataset)
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    load_dotenv()

    args = parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_dir():
        cache_dir.mkdir(parents=True, exist_ok=True)

    main(
        subgraph_fpath=args.subgraph_fpath,
        node_fpath=args.node_fpath,
        edge_fpath=args.edge_fpath,
        model_type=args.model_type,
        model_id=args.model_id,
        cache_dir=cache_dir,
        save_path=args.save_path,
        n_responses=args.n_responses,
        max_quality_retries=args.max_quality_retries,
        match_direction=args.match_direction,
        temperature=args.temperature,
    )
