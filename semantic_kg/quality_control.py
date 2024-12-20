import os
import re
import abc
from typing import Callable, Literal, Optional, Protocol

import spacy
from spacy.cli.download import download
import pandas as pd
from tqdm import tqdm

from semantic_kg import utils
from semantic_kg.models.base import BaseTextGeneration
from semantic_kg.models.llm import InvalidResponseError
from semantic_kg.models.utils import structured_generation_helper


def _node_equality(node1: str, node2: str) -> bool:
    """Default node equality function"""
    return node1.lower() == node2.lower()


class NLPNodeEquality:
    def __init__(self, lemmatize: bool = True, preserve_order: bool = True) -> None:
        """Node equality function that applies NLP processing

        Parameters
        ----------
        lemmatize : bool, optional
            If true then compares lemmatized tokens, by default True
        preserve_order : bool, optional
            If True then order must match, otherwise does bag-of-words comparison,
            by default True
        """
        try:
            self.nlp = spacy.load(os.environ.get("SPACY_MODEL_PATH", "en_core_web_sm"))
        except OSError:
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        self.lemmatize = lemmatize
        if self.lemmatize:
            self.nlp.get_pipe("lemmatizer")
        self.preserve_order = preserve_order

    def __call__(self, node1: str, node2: str) -> bool:
        node1 = node1.lower()
        node2 = node2.lower()

        doc1 = self.nlp(node1)
        doc2 = self.nlp(node2)

        attr = "lemma_" if self.lemmatize else "text"

        tokens1 = [getattr(token, attr) for token in doc1 if not token.is_stop]
        tokens2 = [getattr(token, attr) for token in doc2 if not token.is_stop]

        if self.preserve_order:
            return tokens1 == tokens2
        else:
            return set(tokens1) == set(tokens2)


class TripleCompare:
    def __init__(
        self,
        match_nodes: bool = True,
        match_edges: bool = True,
        match_direction: bool = True,
        node_match_fn: Optional[Callable[[str, str], bool]] = None,
        edge_match_fn: Optional[Callable[[str, str], bool]] = None,
    ) -> None:
        """Class to compare equality between 2 sets of triples

        Parameters
        ----------
        match_nodes : bool, optional
            If True then all nodes must match, by default True
        match_edges : bool, optional
            If True then all edges must match, by default True
        match_direction : bool, optional
            If True then the direction of the triple must match, by default True
        node_match_fn : Optional[Callable[[str, str], bool]], optional
            Function to compare 2 nodes. If none provided then simply checks for
            equality between both lowercase strings, by default None
        edge_match_fn : Optional[Callable[[str, str], bool]], optional
            Function to compare 2 edges, If none provided then simply checks for
            equality between both lowercase strings, by default None

        Raises
        ------
        ValueError
            If both `match_nodes` and `match_edges` is False
        """
        self.match_nodes = match_nodes
        self.match_edges = match_edges
        if not self.match_nodes and not self.match_edges:
            raise ValueError("`match_nodes` and `match_edges` can't both be False")
        self.match_direction = match_direction

        if not node_match_fn:
            self.node_match_fn = _node_equality
        else:
            self.node_match_fn = node_match_fn

        if not edge_match_fn:
            self.edge_match_fn = _node_equality
        else:
            self.edge_match_fn = edge_match_fn

    def _get_data(self, triple_dict: dict[str, str] | str) -> str:
        """Helper method for instance where data is not nested"""
        if isinstance(triple_dict, dict):
            return triple_dict["name"]
        elif isinstance(triple_dict, str):
            return triple_dict
        else:
            raise ValueError(f"Unknown data-type for triple: {type(triple_dict)}")

    def _get_triple_data(
        self, subgraph_triples: list[dict[str, dict[str, str]]]
    ) -> list[dict[str, str]]:
        """Helper function to unpack nested dict"""
        all_triple_data = []
        for triple in subgraph_triples:
            triple_data = {}
            if self.match_nodes:
                triple_data["src"] = self._get_data(triple["source_node"])
            if self.match_edges:
                triple_data["relation"] = self._get_data(triple["relation"])
            if self.match_nodes:
                triple_data["target"] = self._get_data(triple["target_node"])

            all_triple_data.append(triple_data)

        return all_triple_data

    def _triple_match_direction(
        self, triple: dict[str, str], rc_triple: dict[str, str]
    ) -> bool:
        """Checks for a match between triples when direction matters"""
        matches = []
        if self.match_nodes:
            matches.append(self.node_match_fn(triple["src"], rc_triple["src"]))
        if self.match_edges:
            matches.append(
                self.edge_match_fn(triple["relation"], rc_triple["relation"])
            )
        if self.match_nodes:
            matches.append(self.node_match_fn(triple["target"], rc_triple["target"]))

        return all(matches)

    def _triple_match_direction_false(
        self, triple: dict[str, str], rc_triple: dict[str, str]
    ) -> bool:
        """Checks for a match between triples when direction doesn't matter"""
        matches = []
        if self.match_nodes:
            # Compares both orderings of nodes
            same_order = self.node_match_fn(
                triple["src"], rc_triple["src"]
            ) and self.node_match_fn(triple["target"], rc_triple["target"])
            reversed_order = self.node_match_fn(
                triple["src"], rc_triple["target"]
            ) and self.node_match_fn(triple["target"], rc_triple["src"])
            matches.append(same_order or reversed_order)
        if self.match_edges:
            matches.append(
                self.edge_match_fn(triple["relation"], rc_triple["relation"])
            )

        return all(matches)

    def score(
        self,
        original_subgraph: list[dict[str, dict[str, str]]],
        reconstructed_subgraph: list[dict[str, dict[str, str]]],
    ) -> list[bool]:
        """Checks whether 2 subgraphs represented as list of triples match

        Function returns a list of booleans indicating whether a triple in
        `original_subgraph` was also found in `reconstructed_subgraph`

        Parameters
        ----------
        original_subgraph : list[dict[str, dict[str, str]]]
            List of triples representing subgraph
        reconstructed_subgraph : list[dict[str, dict[str, str]]]
            List of triples representing reconstructed subgraph

        Returns
        -------
        list[bool]
            List of booleans indicating a match between a source triple and
            reconstructed triple
        """
        subgraph_data = self._get_triple_data(original_subgraph)
        reconstructed_data = self._get_triple_data(reconstructed_subgraph)

        triple_match_fn = (
            self._triple_match_direction
            if self.match_direction
            else self._triple_match_direction_false
        )

        triple_matches = []
        for triple in subgraph_data:
            if any(
                triple_match_fn(triple, rc_triple) for rc_triple in reconstructed_data
            ):
                triple_matches.append(True)
            else:
                triple_matches.append(False)

        return triple_matches


class ScorerProtocol(Protocol):
    """Protocol class for scorer classes"""

    @abc.abstractmethod
    def score(
        self,
        response: str,
        triples: list[dict[str, dict[str, str]]],
        **kwargs,
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
        scorer: Optional[TripleCompare] = None,
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
            self.scorer = TripleCompare(
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


class BatchReconstructionScorer:
    def __init__(
        self,
        scorer: KGReconstructionScorer,
        triple_field: str = "subgraph_triples",
        perturbed_triple_field: str = "perturbed_triples",
        response_field: str = "original_response",
        perturbed_response_field: str = "perturbed_response",
    ) -> None:
        """Class for applying reconstruction scoring to a dataframe in batch

        Parameters
        ----------
        scorer : KGReconstructionScorer
            Instantiated reconstruction scorer to apply to each response
        triple_field : str, optional
            Field containing original subgraph triple data, by default
            "subgraph_triples"
        perturbed_triple_field : str, optional
            Field containing perturbed subgraph triple data, by default
            "perturbed_triples"
        response_field : str, optional
            Field containing model-generated responses for original
            subgraph, by default "original_response"
        perturbed_response_field : str, optional
            Field containing model-generated responses to pertubed
            subgraph, by default "perturbed_response"
        """
        self.scorer = scorer
        self.triple_field = triple_field
        self.perturbed_triple_field = perturbed_triple_field
        self.response_field = response_field
        self.perturbed_response_field = perturbed_response_field

    def _get_total_responses(self, df: pd.DataFrame) -> int:
        return (
            df[self.response_field].apply(len).sum()
            + df[self.perturbed_response_field].apply(len).sum()
        )

    def _get_scored_responses(
        self,
        triples: list[dict[str, dict[str, str]]],
        responses: list[str],
        pbar: tqdm,
    ) -> tuple[list[bool], list[dict[str, dict[str, str]]]]:
        scored_responses = []
        reconstructed_triples = []
        for response in responses:
            score = self.scorer.score(response, triples)
            scored_responses.append(True if score == 1.0 else False)
            reconstructed_triples.append(self.scorer.reconstructed_triples)
            pbar.update(1)

        return scored_responses, reconstructed_triples

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates a dataset of scores for each response"""
        pbar = tqdm(total=self._get_total_responses(df))

        sg_scores = []
        p_sg_scores = []
        sg_reconstructed_triples = []
        p_sg_reconstructed_triples = []
        for idx, row in df.iterrows():
            triples = row[self.triple_field]

            original_responses = row[self.response_field]
            try:
                sg_responses_scores, r_triples = self._get_scored_responses(
                    triples,
                    original_responses,
                    pbar,
                )
            except InvalidResponseError:
                print(
                    f"Unable to get valid responses for original responses at idx: {idx}"
                )
                sg_responses_scores = None
                r_triples = None

            perturbed_responses = row[self.perturbed_response_field]
            try:
                p_sg_response_scores, r_p_triples = self._get_scored_responses(
                    triples,
                    perturbed_responses,
                    pbar,
                )
            except InvalidResponseError:
                print(
                    f"Unable to get valid responses for perturbed responses at idx: {idx}"
                )
                p_sg_response_scores = None
                r_p_triples = None

            sg_scores.append(sg_responses_scores)
            p_sg_scores.append(p_sg_response_scores)
            sg_reconstructed_triples.append(r_triples)
            p_sg_reconstructed_triples.append(r_p_triples)

        df["original_response_scores"] = sg_scores
        df["perturbed_response_scores"] = p_sg_scores
        df["reconstructed_triples"] = sg_reconstructed_triples
        df["reconstructed_perturbed_triples"] = p_sg_reconstructed_triples

        return df
