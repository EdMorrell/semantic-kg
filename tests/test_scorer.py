import pytest

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
