import os
from typing import Callable, Optional

import spacy
from spacy.cli.download import download


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
