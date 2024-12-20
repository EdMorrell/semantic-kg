import pandas as pd
from tqdm import tqdm

from semantic_kg.quality_control import scorer
from semantic_kg.models.base import BaseTextGeneration
from semantic_kg.models.llm import InvalidResponseError
from semantic_kg.quality_control import graph_reconstruction


def build_reconstruction_scorer(
    scoring_model: BaseTextGeneration, match_direction: bool = True
) -> scorer.KGReconstructionScorer:
    """Helper function to load an instantiated reconstruction scorer

    Parameters
    ----------
    scoring_model : BaseTextGeneration
        Scoring model to use for reconstruction
    match_direction : bool, optional
        If True then the direction of the triple must match, by default True
    """
    node_checker = graph_reconstruction.NLPNodeEquality(preserve_order=False)
    triple_compare = graph_reconstruction.TripleCompare(
        node_match_fn=node_checker,
        edge_match_fn=node_checker,
        match_direction=match_direction,
    )

    user_prompt_template = """Statement: {statement}"""

    reconstruction_scorer = scorer.KGReconstructionScorer(
        llm=scoring_model,
        prompt_template=user_prompt_template,
        max_retries=5,
        scorer=triple_compare,
    )

    return reconstruction_scorer


class BatchReconstructionScorer:
    def __init__(
        self,
        scorer: scorer.KGReconstructionScorer,
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
