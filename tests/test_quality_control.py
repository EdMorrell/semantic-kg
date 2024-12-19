import pytest

from semantic_kg import quality_control


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


class TestKGReconstructioncompare:
    def test_compare_match_all_direction(
        self,
        test_triples: list[dict[str, dict[str, str]]],
        test_reconstructed_triples: list[dict[str, dict[str, str]]],
    ) -> None:
        compare = quality_control.TripleCompare(
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
        compare = quality_control.TripleCompare(
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
        compare = quality_control.TripleCompare(
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
        compare = quality_control.TripleCompare(
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


class TestRegexMatcherScorer:
    @pytest.mark.parametrize(
        "input_string, expected_output",
        [
            # Single quotes
            ("Phenotypes such as 'positive regulation of complement activation'", 1.0),
            # Double quotes
            ('Phenotypes such as "positive regulation of complement activation"', 1.0),
            # Backticks
            ("Phenotypes such as `positive regulation of complement activation`", 1.0),
            # No quotes
            ("Phenotypes such as positive regulation of complement activation", 0.0),
            # Multiple quoted strings
            (
                '"Regulation of complement activation" is a parent-child of "positive regulation of complement activation"',
                1.0,
            ),
        ],
    )
    def test_quote_match(self, input_string: str, expected_output: float) -> None:
        scorer = quality_control.RegexMatcherScorer(
            pattern=quality_control.QUOTE_REGEX, mode="exists"
        )
        assert scorer.score(input_string) == expected_output

    @pytest.mark.parametrize(
        "input_string, expected_output",
        [
            # Single quotes
            ("Phenotypes such as 'positive regulation of complement activation'", 0.0),
            # Double quotes
            ('Phenotypes such as "positive regulation of complement activation"', 0.0),
            # No quotes
            ("Phenotypes such as positive regulation of complement activation", 1.0),
            # Multiple quoted strings
            (
                '"Regulation of complement activation" is a parent-child of "positive regulation of complement activation"',
                0.0,
            ),
        ],
    )
    def test_quote_match_not_exists(
        self, input_string: str, expected_output: float
    ) -> None:
        scorer = quality_control.RegexMatcherScorer(
            pattern=quality_control.QUOTE_REGEX, mode="not_exists"
        )

        assert scorer.score(input_string) == expected_output
