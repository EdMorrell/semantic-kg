import re
import abc
from typing import Literal, Optional, Protocol

from semantic_kg import utils
from semantic_kg.models.base import BaseTextGeneration
from semantic_kg.quality_control import graph_reconstruction
from semantic_kg.models.utils import structured_generation_helper


class ScorerProtocol(Protocol):
    """Protocol class for scorer classes"""

    @abc.abstractmethod
    def score(
        self,
        response: str,
        triples: list[dict[str, dict[str, str]]],
    ) -> float:
        # TODO: Figure out how to avoid relying on `triples` input
        pass


# Regex pattern to match quoted strings
QUOTE_REGEX = r"[\"'`]([^\"'`]*)[\"'`]"


class RegexMatcherScorer:
    def __init__(
        self, pattern: str, mode: Literal["exists", "not_exists"] = "exists"
    ) -> None:
        """Scorer class to match a regex pattern in a response

        Parameters
        ----------
        pattern : str
            Regex pattern to match in response
        mode : Literal["exists", "not_exists"], optional
            Operating modes:
            - `exists`: If pattern exists in response then score is 1.0
            - `not_exists`: If pattern doesn't exist in response then score is 1.0
        """
        self.pattern = pattern
        self.mode = mode

    def score(
        self, response: str, triples: list[dict[str, dict[str, str]]] = None
    ) -> float:
        """Scores a response based on the presence of a regex pattern"""
        pattern_found = re.search(self.pattern, response)

        if self.mode == "exists":
            return 1.0 if pattern_found else 0.0
        elif self.mode == "not_exists":
            return 0.0 if pattern_found else 1.0
        else:
            raise ValueError("Invalid mode. Mode should be 'exists' or 'not_exists'.")


class KGReconstructionScorer:
    def __init__(
        self,
        llm: BaseTextGeneration,
        prompt_template: str,
        scorer: Optional[graph_reconstruction.TripleCompare] = None,
        max_retries: int = 3,
        max_tokens: int = 1000,
        increase_tokens_on_retry: bool = True,
        seed: int = 42,
    ) -> None:
        """Scorer reconstructs subgraph from response to check validity

        This scorer class uses an LLM to reconstruct a subgraph from the
        generated subgraph and scores the response based on the ability to
        reconstruct the subgraph

        Parameters
        ----------
        llm : BaseTextGeneration
            LLM to use for reconstruction
        prompt_template : str
            Prompt template to use for instructing scorer LLM. Must contain a field
            called `statement`
        scorer : Optional[TripleCompare], optional
            An instance of `TripleCompare` to compare original and reconstructed
            triples, If None provided then automatically instantiates. By default None
        max_retries : int, optional
            Maximum number of retries for subgraph reconstruction, by default 3
        max_tokens : int, optional
            Maximum number of tokens for scorer LLM to return, by default 1000
        increase_tokens_on_retry : bool, optional
            If True then for every failure, `max_tokens` is iterated by 500, Useful
            when encountering parsing errors for larger subgraphs, by default True
        seed : int, optional
            Random-seed for reproducibility, by default 42

        Raises
        ------
        ValueError
            If prompt template doesn't contain a `statement` placeholder
        """
        self.llm = llm

        self.prompt_template = prompt_template
        if "statement" not in utils.find_field_placeholders(self.prompt_template):
            raise ValueError("Prompt template must contain a field for `statement`")

        if not scorer:
            self.scorer = graph_reconstruction.TripleCompare(
                match_nodes=True, match_edges=True, match_direction=True
            )
        else:
            self.scorer = scorer

        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.increase_tokens_on_retry = increase_tokens_on_retry
        self.seed = seed

    def score(
        self,
        response: str,
        triples: list[dict[str, dict[str, str]]],
    ) -> float:
        """Reconstructs a subgraph from `response` and compares to original

        Method takes a model-generated response and instructs an LLM to compare
        to the original list of triples to score quality. The quality score is
        a float indicating the proportion of triples in the original subgraph
        the scorer LLM was able to recreate. A score of 1.0 indicates perfect
        reproduction.

        Parameters
        ----------
        response : str
            Model-generated response
        triples : list[dict[str, dict[str, str]]]
            Original subgraph triples

        Returns
        -------
        float
            A float indicating the proportion of correctly reconstructed triples

        Raises
        ------
        InvalidResponseError
            If the scoring model is unable to generate a valid subgraph after
            `self.max_retries` retries.
        """
        prompt = self.prompt_template.format(statement=response)

        scorer_response = structured_generation_helper(
            self.llm,
            prompt,
            self.max_retries,
            self.max_tokens,
            self.seed,
            self.increase_tokens_on_retry,
        )
        reconstructed_triples = scorer_response["triples"]

        scores = self.scorer.score(triples, reconstructed_triples)

        # Logs reconstructed triples as attribute for debugging purposes
        self.reconstructed_triples = reconstructed_triples

        return sum(scores) / len(scores)
