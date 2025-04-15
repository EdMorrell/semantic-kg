import functools
from pathlib import Path
import random
from typing import Callable, Literal, Optional, Type, TypedDict

import openai
import backoff
import pandas as pd
from tqdm import tqdm
from openai import BadRequestError

from semantic_kg import utils
from semantic_kg.sampling import SubgraphSampler, SubgraphDataset
from semantic_kg.perturbation import GraphPerturber, build_perturber
from semantic_kg.datasets import (
    KGLoader,
    create_edge_map,
    get_valid_node_pairs,
    EDGE_MAPPING_TYPE,
)
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


class SubgraphPipeline:
    def __init__(
        self,
        src_node_id_field: str,
        src_node_type_field: str,
        src_node_name_field: str,
        edge_name_field: str,
        target_node_id_field: str,
        target_node_type_field: str,
        target_node_name_field: str,
        directed: bool,
        save_path: Optional[Path | str] = None,
        edge_map: Optional[EDGE_MAPPING_TYPE] = None,
        replace_map: Optional[EDGE_MAPPING_TYPE] = None,
        valid_node_pairs: Optional[list[tuple[str, str]]] = None,
        edge_addition: bool = True,
        edge_deletion: bool = True,
        edge_replacement: bool = True,
        node_removal: bool = True,
        p_prob: Optional[list[float]] = None,
        p_perturbation_range: Optional[tuple[float, float]] = None,
        kg_loader: Optional[KGLoader] = None,
        perturber: Optional[GraphPerturber] = None,
    ) -> None:
        """Pipeline for generating pairs of subgraph and perturbed subgraphs
        by sampling from a Knowledge-Graph

        Parameters
        ----------
        src_node_id_field : str
            Name of ID field for source-nodes in knowledge-graph
        src_node_type_field : str
            Name of type field for source-nodes in knowledge-graph
        src_node_name_field : str
            Field indicating display name of source-nodes in knowledge-graph
        edge_name_field : str
            Field indicating display name of edges in knowledge-graph
        target_node_id_field : str
            Name of ID field for target-nodes in knowledge-graph
        target_node_type_field : str
            Name of type field for target-nodes in knowledge-graph
        target_node_name_field : str
            Field indicating display name of target-nodes in knowledge-graph
        directed : bool
            Whether or not the knowledge-graph is stored in directed setting
        save_path : Path | str, optional
            Path to directory to save final dataset
        edge_map : Optional[EDGE_MAPPING_TYPE], optional
            Map from node-pair types to edge-names (
            e.g. ("PROTEIN", "PROTEIN"): ["protein-protein-interaction"]), If not
            provided will be inferred automatically from the `triple_df`. By default
            None
        replace_map : Optional[EDGE_MAPPING_TYPE], optional
            Map denoting accepted replacement values for edges during a replacement
            perturbation. Replacement mappings are denoted by a mapping from node-type
            to valid edge-values. For example: {
                ("DRUG", "PROTEIN"): ["activates", "inhibits"]
            }
            If not provided will be inferred automatically from `triple_df`. By default
            None
        valid_node_pairs : Optional[list[tuple[str, str]]], optional
            A list denoting valid node-type pairs, (e.g. [("PROTEIN", "PROTEIN"),
            ("PROTEIN", "GENE"), ...). If not provided will be inferred automatically
            from `triple_df`. By default None
        edge_addition : bool, optional
            Whether to perform edge-addition perturbations. By default True
        edge_deletion : bool, optional
            Whether to perform an edge-deletion perturbations. By default True
        edge_replacement : bool, optional
            Whether to perform edge-replacement perturbations. By default True
        node_removal : bool, optional
            Whether to perform node-removal perturbations. By default True
        p_prob : list[float], optional
            List of floats denoting the probability of applying a given perturbation.
            Default order: [p_edge_addition, p_edge_deletion, p_edge_replacement,
            p_node_removal]
            If not provided denotes to default of [0.3, 0.3, 0.3, 0.1] (such that node
            removal perturbations are slightly less probable.)
        p_perturbation_range : Optional[tuple[float, float]], optional
            Optionally specify the proportion of perturbations to apply, relative to the
            number of nodes, by default None.
        kg_loader : Optional[KGLoader], optional
            Optionally provide an instance of KGLoader to load a knowledge-graph from
            `triple_df`, by default None
        perturber : Optional[GraphPerturber], optional
            Optionally provide an instance of GraphPerturber, to perturb individual
            subgraphs, by default None
        """
        self.src_node_id_field = src_node_id_field
        self.src_node_type_field = src_node_type_field
        self.src_node_name_field = src_node_name_field
        self.edge_name_field = edge_name_field
        self.target_node_id_field = target_node_id_field
        self.target_node_type_field = target_node_type_field
        self.target_node_name_field = target_node_name_field

        self.directed = directed

        self.save_path = save_path
        if self.save_path:
            self.save_path = Path(self.save_path)

        self.edge_map = edge_map
        self.replace_map = replace_map
        self.valid_node_pairs = valid_node_pairs

        self.edge_addition = edge_addition
        self.edge_deletion = edge_deletion
        self.edge_replacement = edge_replacement
        self.node_removal = node_removal
        self.p_prob = p_prob
        self.p_perturbation_range = p_perturbation_range

        if not kg_loader:
            self.kg_loader = KGLoader(
                src_node_id_field=self.src_node_id_field,
                src_node_type_field=self.src_node_type_field,
                src_node_name_field=self.src_node_name_field,
                edge_name_field=self.edge_name_field,
                target_node_id_field=self.target_node_id_field,
                target_node_type_field=self.target_node_type_field,
                target_node_name_field=self.target_node_name_field,
            )
        else:
            self.kg_loader = kg_loader

        self.perturber = perturber
        if not self.perturber:
            if (
                not self.edge_addition
                and not self.edge_deletion
                and not self.edge_replacement
                and not self.node_removal
            ):
                raise ValueError("Perturbation options can't all be False")

    def generate(
        self,
        triple_df: pd.DataFrame,
        n_iter: int = 1000,
        n_node_range: tuple[int, int] = (3, 10),
        sample_method: Literal["bfs", "bfs_node_diversity"] = "bfs_node_diversity",
        start_node_types: Optional[list[str]] = None,
        save_subgraphs: bool = False,
        max_retries: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generates a dataset of subgraph/perturbed subgraph pairs

        Method takes a triple dataframe and uses it to sample a subgraph dataset

        Parameters
        ----------
        triple_df : pd.DataFrame
            Dataframe consisting of knowledge-graph triples (subject, predicate, object)
        n_iter : int, optional
            Number of iterations (no. of final rows in dataset), by default 1000
        n_node_range : tuple[int, int], optional
            A range indicating the number of nodes in each sampled subgraph, by default
            (3, 10)
        sample_method : Literal["bfs", "bfs_node_diversity"], optional
           Method to sample subgraph, by default "bfs_node_diversity"
        start_node_types : Optional[list[str]], optional
            A list of node-types to start a sampled subgraph from. Defaults to all possible
            node-types.
        save_subgraphs : bool, optional
            If True then will save intermediate subgraphs. This will significantly
            slow execution, by default False

        Returns
        -------
        pd.DataFrame
            Final subgraph dataframe
        """
        g = self.kg_loader.load(triple_df=triple_df, directed=self.directed)

        if not self.edge_map:
            self.edge_map = create_edge_map(
                triple_df,
                src_node_type_field=self.kg_loader.src_node_type_field,
                target_node_type_field=self.kg_loader.target_node_type_field,
                edge_name_field=self.kg_loader.edge_name_field,
                directed=self.directed,
            )
        if not self.replace_map:
            self.replace_map = {k: v for k, v in self.edge_map.items() if len(v) > 1}

        if not self.valid_node_pairs:
            self.valid_node_pairs = get_valid_node_pairs(
                triple_df,
                src_node_type_field=self.kg_loader.src_node_type_field,
                target_node_type_field=self.kg_loader.target_node_type_field,
            )

        if not self.perturber:
            self.perturber = build_perturber(
                edge_map=self.edge_map,
                valid_node_pairs=self.valid_node_pairs,  # type: ignore
                replace_map=self.replace_map,
                directed=self.directed,
                edge_addition=self.edge_addition,
                edge_deletion=self.edge_deletion,
                edge_replacement=self.edge_replacement,
                node_removal=self.node_removal,
                p_prob=self.p_prob,
            )

        sampler = SubgraphSampler(graph=g, method=sample_method)

        kwargs = {}
        if self.p_perturbation_range is not None:
            kwargs["p_perturbation_range"] = self.p_perturbation_range

        if start_node_types is not None:
            kwargs["start_node_attrs"] = {"node_type": start_node_types}

        subgraph = SubgraphDataset(
            graph=g,
            subgraph_sampler=sampler,
            perturber=self.perturber,
            n_node_range=n_node_range,
            save_subgraphs=save_subgraphs,
            dataset_save_dir=self.save_path,  # type: ignore
            **kwargs,  # type: ignore
        )
        if not max_retries:
            max_retries = n_iter * 10
        sample_df = subgraph.generate(n_iter, max_retries)

        return sample_df


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
        total = len(subgraph_dataset) * self.n_responses * 2
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

        # Updates the progress bar with missing responses
        pbar.update(self.n_responses - total_responses)

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
