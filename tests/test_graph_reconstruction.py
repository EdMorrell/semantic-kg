import pytest

from semantic_kg.quality_control import graph_reconstruction


@pytest.fixture
def test_triples() -> list[dict[str, dict[str, str]]]:
    return [
        {
            "source_node": {"name": "HSPBP1"},
            "relation": {"name": "ppi"},
            "target_node": {"name": "APP"},
        },
        {
            "source_node": {"name": "APP"},
            "relation": {"name": "ppi"},
            "target_node": {"name": "TCP10L"},
        },
        {
            "source_node": {"name": "APP"},
            "relation": {"name": "ppi"},
            "target_node": {"name": "PCM1"},
        },
        {
            "source_node": {"name": "APP"},
            "relation": {"name": "linked to"},
            "target_node": {"name": "cellular response to norepinephrine stimulus"},
        },
        {
            "source_node": {"name": "response to norepinephrine"},
            "relation": {"name": "parent-child"},
            "target_node": {"name": "response to catecholamine"},
        },
        {
            "source_node": {"name": "PPARGC1A"},
            "relation": {"name": "interacts with"},
            "target_node": {"name": "response to norepinephrine"},
        },
    ]


@pytest.fixture
def test_reconstructed_triples() -> list[dict[str, dict[str, str]]]:
    return [
        # Match to 0
        {
            "source_node": {"name": "HSPBP1"},
            "relation": {"name": "ppi"},
            "target_node": {"name": "APP"},
        },
        # Node-Match to 1
        {
            "source_node": {"name": "APP"},
            "relation": {"name": "null"},
            "target_node": {"name": "TCP10L"},
        },
        # Match to 2 if ignoring direction
        {
            "source_node": {"name": "PCM1"},
            "relation": {"name": "ppi"},
            "target_node": {"name": "APP"},
        },
        # Node match to 3 if ignoring direction
        {
            "source_node": {"name": "cellular response to norepinephrine stimulus"},
            "relation": {"name": "interacts with"},
            "target_node": {"name": "APP"},
        },
        # Edge match
        {
            "source_node": {"name": "cellular response to norepinephrine stimulus"},
            "relation": {"name": "parent-child"},
            "target_node": {"name": "cellular response to catecholamine stimulus"},
        },
        # No match
        {
            "source_node": {"name": "GDF12A"},
            "relation": {"name": "linked to"},
            "target_node": {"name": "HBDAS2"},
        },
    ]


class TestKGReconstructionCompare:
    def test_compare_match_all_direction(
        self,
        test_triples: list[dict[str, dict[str, str]]],
        test_reconstructed_triples: list[dict[str, dict[str, str]]],
    ) -> None:
        compare = graph_reconstruction.TripleCompare(
            match_nodes=True, match_edges=True, match_direction=True
        )

        assert compare.score(test_triples, test_reconstructed_triples) == [
            True,
            False,
            False,
            False,
            False,
            False,
        ]

    def test_compare_node_match_direction(
        self,
        test_triples: list[dict[str, dict[str, str]]],
        test_reconstructed_triples: list[dict[str, dict[str, str]]],
    ) -> None:
        compare = graph_reconstruction.TripleCompare(
            match_nodes=True, match_edges=False, match_direction=True
        )

        assert compare.score(test_triples, test_reconstructed_triples) == [
            True,
            True,
            False,
            False,
            False,
            False,
        ]

    def test_compare_match_all_no_direction(
        self,
        test_triples: list[dict[str, dict[str, str]]],
        test_reconstructed_triples: list[dict[str, dict[str, str]]],
    ) -> None:
        compare = graph_reconstruction.TripleCompare(
            match_nodes=True, match_edges=True, match_direction=False
        )

        assert compare.score(test_triples, test_reconstructed_triples) == [
            True,
            False,
            True,
            False,
            False,
            False,
        ]

    def test_compare_node_match_no_direction(
        self,
        test_triples: list[dict[str, dict[str, str]]],
        test_reconstructed_triples: list[dict[str, dict[str, str]]],
    ) -> None:
        compare = graph_reconstruction.TripleCompare(
            match_nodes=True, match_edges=False, match_direction=False
        )

        assert compare.score(test_triples, test_reconstructed_triples) == [
            True,
            True,
            True,
            True,
            False,
            False,
        ]
