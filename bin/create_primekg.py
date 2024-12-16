import argparse
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from semantic_kg import models
from semantic_kg.datasets.prime_kg import format_primekg_prompt
from semantic_kg.generation import NLGenerationPipeline, default_exception_selector


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
        "--temperature", type=float, help="Temperature to use during generation"
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        help="Path to save final dataset",
        default=ROOT_DIR / "datasets" / "prime_kg" / "prime_kg_dataset.csv",
    )

    return parser.parse_args()


def main(
    subgraph_fpath: Path,
    model_type: str,
    model_id: str,
    cache_dir: Path,
    save_path: Path,
    n_responses: int = 10,
    temperature: float = 1.0,
) -> None:
    """Script to generate responses from an LLM to PrimeKG subgraphs"""
    subgraph_dataset = pd.read_csv(subgraph_fpath)

    model = models.load_model(
        model_type, model_id=model_id, cache_dir=cache_dir, temperature=temperature
    )
    prompt_template = format_primekg_prompt()

    pipeline = NLGenerationPipeline(
        model,
        prompt_template,
        n_responses,
        backoff_exceptions=default_exception_selector(model_type),  # type: ignore
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
        model_type=args.model_type,
        model_id=args.model_id,
        cache_dir=cache_dir,
        save_path=args.save_path,
        n_responses=args.n_responses,
        temperature=args.temperature,
    )
