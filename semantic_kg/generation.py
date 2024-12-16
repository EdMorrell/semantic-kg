import functools
from typing import Callable, Literal, Optional, Type

import openai
import backoff
import pandas as pd
from tqdm import tqdm

from semantic_kg import utils
from semantic_kg.models.base import BaseTextGeneration


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


class NLGenerationPipeline:
    def __init__(
        self,
        model: BaseTextGeneration,
        prompt_template: str,
        n_responses: int = 10,
        subgraph_field: str = "subgraph_triples",
        pertrubed_subgraph_field: str = "perturbed_subgraph_triples",
        max_tokens: int = 500,
        backoff_exceptions: Optional[list[Type[Exception]]] = None,
        max_backoff_tries: int = 3,
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
        self.subgraph_field = subgraph_field
        self.perturbed_subgraph_field = pertrubed_subgraph_field
        self.max_tokens = max_tokens

        if backoff_exceptions:
            self.backoff_decorator = create_backoff_decorator(
                backoff_exceptions, max_backoff_tries
            )

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
        subgraph_responses = []
        perturbed_subgraph_responses = []
        for _, row in tqdm(
            subgraph_dataset.iterrows(), total=subgraph_dataset.shape[0]
        ):
            if self.backoff_decorator:
                request_func = functools.wraps(self.model.generate)(
                    self.backoff_decorator
                )(self.model.generate)
            else:
                request_func = self.model.generate

            subgraph_prompt = self.prompt_template.format(
                triples=row[self.subgraph_field]
            )
            sg_response = request_func(
                subgraph_prompt, self.n_responses, max_tokens=self.max_tokens
            )
            subgraph_responses.append(sg_response)

            perturbed_subgraph_prompt = self.prompt_template.format(
                triples=row[self.perturbed_subgraph_field]
            )
            p_sg_response = request_func(
                perturbed_subgraph_prompt, self.n_responses, max_tokens=self.max_tokens
            )
            perturbed_subgraph_responses.append(p_sg_response)

        subgraph_dataset["original_response"] = subgraph_responses
        subgraph_dataset["perturbed_response"] = perturbed_subgraph_responses

        return subgraph_dataset
