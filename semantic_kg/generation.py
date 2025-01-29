import functools
import random
from typing import Callable, Literal, Optional, Type, TypedDict

import openai
import backoff
import pandas as pd
from tqdm import tqdm
from openai import BadRequestError

from semantic_kg import utils
from semantic_kg.quality_control.scorer import ScorerProtocol
from semantic_kg.models.base import BaseTextGeneration
from semantic_kg.models.llm import InvalidResponseError


DEFAULT_EXCEPTION_MAP = {
    "openai": [
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
    ]
}


def create_backoff_decorator(
    exceptions: list[Type[Exception]],
    max_tries: int = 3,
) -> Callable[[Callable], Callable]:
    """Creates an instance of a backoff decorator

    Creates a backoff decorator, used to halt API calls when encountering
    an error before retrying

    Parameters
    ----------
    exceptions : list[Type[Exception]]
        Exceptions to include in backoff
    max_tries : int, optional
        Max tried before giving up, by default 3

    Returns
    -------
    Callable[[Callable], Callable]
        Decorator function
    """
    return backoff.on_exception(
        wait_gen=backoff.constant,
        exception=exceptions,
        max_tries=max_tries,
        raise_on_giveup=True,
        jitter=None,
    )


def default_exception_selector(model_type: Literal["openai"]) -> list[Type[Exception]]:
    """Helper function for selecting default backoff exceptions"""
    return DEFAULT_EXCEPTION_MAP[model_type]


class ScorerConfig(TypedDict):
    scorer: ScorerProtocol
    accept_threshold: float


class NLGenerationPipeline:
    def __init__(
        self,
        model: BaseTextGeneration,
        prompt_template: str,
        n_responses: int = 10,
        quality_scorers: Optional[list[ScorerConfig]] = None,
        retry_unacceptable: bool = True,
        max_quality_retries: int = 3,
        subgraph_field: str = "subgraph_triples",
        perturbed_subgraph_field: str = "perturbed_subgraph_triples",
        max_tokens: int = 500,
        backoff_exceptions: Optional[list[Type[Exception]]] = None,
        max_backoff_tries: int = 3,
        seed: int = 42,
    ) -> None:
        """Class for generating natural-language responses to subgraphs

        Parameters
        ----------
        model : BaseTextGeneration
            Model to generate responses with
        prompt_template : str
            Prompt template. Must have placeholder called `triples`
        n_responses : int, optional
            Number of responses per subgraph, by default 10
        quality_scorers : Optional[list[ScorerConfig]], optional
            If provided will score responses and only keep those above
            the accept threshold, by default None
        retry_unacceptable : bool, optional
            If True, will retry generating responses if none are
            acceptable, by default True
        max_quality_retries : int, optional
            Maximum number of retries if no acceptable responses,
            by default 3
        subgraph_field : str, optional
            Field in dataframe containng subgraph, by default
            "subgraph_triples"
        pertrubed_subgraph_field : str, optional
           Field in dataframe containing perturbed subgraphs, by default
           "perturbed_subgraph_triples"
        max_tokens : int, optional
            Max tokens in output, by default 500
        backoff_exceptions : Optional[list[Type[Exception]]], optional
            List of API errors that if encountered the backoff decorator
            will retry before waiting. If none provided then, no backoff,
            by default None
        max_backoff_tries : int, optional
           Maximum number of retries if encountering errors, by default 3
        seed : int, optional
           Random seed for generation, by default 42

        Raises
        ------
        ValueError
            If no field called `triples` found in prompt template
        """
        self.model = model
        self.prompt_template = prompt_template
        if "triples" not in utils.find_field_placeholders(self.prompt_template):
            raise ValueError("Prompt template must contain a field for `triples`")

        self.n_responses = n_responses

        # Configure quality control
        self.quality_scorers = quality_scorers
        if self.quality_scorers:
            self.retry_unacceptable = retry_unacceptable
            self.max_quality_retries = max_quality_retries

        self.subgraph_field = subgraph_field
        self.perturbed_subgraph_field = perturbed_subgraph_field
        self.max_tokens = max_tokens
        self.seed = seed

        if backoff_exceptions:
            self.backoff_decorator = create_backoff_decorator(
                backoff_exceptions, max_backoff_tries
            )
        else:
            self.backoff_decorator = None

        if self.backoff_decorator:
            self.request_func = functools.wraps(self.model.generate)(
                self.backoff_decorator
            )(self.model.generate)
        else:
            self.request_func = self.model.generate

    def _get_pbar(self, subgraph_dataset: pd.DataFrame) -> tqdm:
        total = len(subgraph_dataset) * self.n_responses
        pbar = tqdm(total=total)
        return pbar

    def _generate_without_qc(
        self, triples: list[dict[str, dict[str, str]]], pbar: tqdm
    ) -> list[str]:
        prompt = self.prompt_template.format(triples=triples)
        response = self.request_func(
            prompt, self.n_responses, max_tokens=self.max_tokens
        )
        pbar.update(self.n_responses)
        return response  # type: ignore

    def _qc_generate_with_retry(
        self, triples: list[dict[str, dict[str, str]]], pbar: tqdm
    ) -> list[str]:
        prng = random.Random(self.seed)
        acceptable_responses = []
        total_responses = 0
        retries = 0
        while total_responses < self.n_responses and retries < self.max_quality_retries:
            _seed = prng.randint(0, int(1e12))
            prompt = self.prompt_template.format(triples=triples)
            response = self.request_func(
                prompt, 1, max_tokens=self.max_tokens, seed=_seed
            )
            response = response[0]
            if not response:
                retries += 1
                continue

            passes_all_checks = True
            for qc_checker in self.quality_scorers:  # type: ignore
                try:
                    score = qc_checker["scorer"].score(response, triples)
                except (
                    KeyError,
                    TypeError,
                    InvalidResponseError,
                    BadRequestError,
                ) as err:
                    print(f"Encountered following error for scorer: {err}")
                    passes_all_checks = False
                    break

                if score < qc_checker["accept_threshold"]:
                    passes_all_checks = False
                    break

            if passes_all_checks:
                acceptable_responses.append(response)
                total_responses += 1
                pbar.update(1)
            else:
                retries += 1

        if (
            retries == self.max_quality_retries
            and total_responses < self.n_responses
            and total_responses > 0
        ):
            print(
                f"Only {total_responses} acceptable responses found for {retries} retries"
            )

        if total_responses == 0:
            print("No acceptable responses found")
            acceptable_responses = []

        return acceptable_responses

    def _qc_generate_without_retry(
        self, triples: list[dict[str, dict[str, str]]], pbar: tqdm
    ) -> list[str]:
        responses = self._generate_without_qc(triples, pbar)
        acceptable_responses = []
        for response in responses:
            passes_all_checks = True
            for qc_checker in self.quality_scorers:  # type: ignore
                if (
                    qc_checker["scorer"].score(response, triples)
                    < qc_checker["accept_threshold"]
                ):
                    passes_all_checks = False
                    break

            if passes_all_checks:
                acceptable_responses.append(response)

        return acceptable_responses

    def _generate_with_qc(
        self, triples: list[dict[str, dict[str, str]]], pbar: tqdm
    ) -> list[str]:
        if self.retry_unacceptable:
            return self._qc_generate_with_retry(triples, pbar)
        return self._qc_generate_without_retry(triples, pbar)

    def generate(self, subgraph_dataset: pd.DataFrame) -> pd.DataFrame:
        """Generates a `subgraph_dataset` with natural-language responses

        Parameters
        ----------
        subgraph_dataset : pd.DataFrame
            Dataset containing subgraph and perturbed subgraph triples

        Returns
        -------
        pd.DataFrame
            Dataframe with added natural-language statements
        """
        pbar = self._get_pbar(subgraph_dataset)

        subgraph_responses = []
        perturbed_subgraph_responses = []
        for _, row in subgraph_dataset.iterrows():
            subgraph_triples = row[self.subgraph_field]
            if not self.quality_scorers:
                sg_response = self._generate_without_qc(subgraph_triples, pbar)
            else:
                sg_response = self._generate_with_qc(subgraph_triples, pbar)
            subgraph_responses.append(sg_response)

            perturbed_subgraph_triples = row[self.perturbed_subgraph_field]
            if not self.quality_scorers:
                p_sg_response = self._generate_without_qc(
                    perturbed_subgraph_triples, pbar
                )
            else:
                p_sg_response = self._generate_with_qc(perturbed_subgraph_triples, pbar)
            perturbed_subgraph_responses.append(p_sg_response)

        subgraph_dataset["original_response"] = subgraph_responses
        subgraph_dataset["perturbed_response"] = perturbed_subgraph_responses

        return subgraph_dataset
