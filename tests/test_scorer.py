import copy
import pytest

from semantic_kg.models.base import BaseTextGeneration
from semantic_kg.quality_control import scorer


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
        regex_scorer = scorer.RegexMatcherScorer(
            pattern=scorer.QUOTE_REGEX, mode="exists"
        )
        assert regex_scorer.score(input_string) == expected_output

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
        regex_scorer = scorer.RegexMatcherScorer(
            pattern=scorer.QUOTE_REGEX, mode="not_exists"
        )

        assert regex_scorer.score(input_string) == expected_output


class MockLLM(BaseTextGeneration):
    def __init__(
        self,
        return_triples: list[dict[str, dict[str, str]]],
    ) -> None:
        self.return_triples = return_triples

    def generate(
        self, prompt: str, n_responses: int, max_tokens: int, seed=None
    ) -> list[str | None]:
        return [str({"triples": self.return_triples})]


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
    ]


class TestKGReconstructionScorer:
    def test_score(self, test_triples: list[dict[str, dict[str, str]]]) -> None:
        mock_llm = MockLLM(return_triples=test_triples)

        kg_reconstruction_scorer = scorer.KGReconstructionScorer(
            llm=mock_llm,
            prompt_template="{statement}",
            scorer=None,
        )

        assert kg_reconstruction_scorer.score("", test_triples) == 1.0

    def test_score_incorrect_triple(
        self, test_triples: list[dict[str, dict[str, str]]]
    ) -> None:
        rtn_triples = copy.deepcopy(test_triples)
        rtn_triples[-1]["source_node"]["name"] = "GDF12A"

        mock_llm = MockLLM(return_triples=rtn_triples)

        kg_reconstruction_scorer = scorer.KGReconstructionScorer(
            llm=mock_llm,
            prompt_template="{statement}",
            scorer=None,
        )

        assert kg_reconstruction_scorer.score("", test_triples) == 0.0

    def test_score_missing_triple(
        self, test_triples: list[dict[str, dict[str, str]]]
    ) -> None:
        rtn_triples = copy.deepcopy(test_triples)
        del rtn_triples[-1]

        mock_llm = MockLLM(return_triples=rtn_triples)

        kg_reconstruction_scorer = scorer.KGReconstructionScorer(
            llm=mock_llm,
            prompt_template="{statement}",
            scorer=None,
        )

        assert kg_reconstruction_scorer.score("", test_triples) == 0.0

    def test_score_extra_triple(
        self, test_triples: list[dict[str, dict[str, str]]]
    ) -> None:
        rtn_triples = copy.deepcopy(test_triples)
        rtn_triples.append(
            {
                "source_node": {"name": "GDF12A"},
                "relation": {"name": "linked to"},
                "target_node": {"name": "HBDAS2"},
            }
        )

        mock_llm = MockLLM(return_triples=rtn_triples)

        kg_reconstruction_scorer = scorer.KGReconstructionScorer(
            llm=mock_llm,
            prompt_template="{statement}",
            scorer=None,
        )

        assert kg_reconstruction_scorer.score("", test_triples) == 0.0
