import ast
import random
from typing import Optional


from semantic_kg.models.base import BaseTextGeneration
from semantic_kg.models import llm


def structured_generation_helper(
    model: BaseTextGeneration,
    prompt: str,
    max_retries: int,
    max_tokens: int,
    seed: Optional[int],
    increase_tokens_on_retry: bool,
) -> dict:
    """Helper function for retrying invalid requests when using structured generation"""
    if not seed:
        seed = 42
    prng = random.Random(seed)
    attempt = 0
    max_tokens = max_tokens
    while attempt < max_retries:
        _seed = prng.randint(0, int(1e12))
        response = model.generate(prompt, 1, max_tokens, seed=_seed)
        try:
            response = ast.literal_eval(response[0])  # type: ignore
            break
        except SyntaxError as err:
            print(f"Error {err}. Retries left: {max_retries - attempt - 1}")
            prompt = (
                f"{prompt}\n. You got the following error: {err}."
                f"Please try again with valid JSON"
            )
            attempt += 1
            # Increases `max_tokens` for next retry
            if increase_tokens_on_retry:
                max_tokens += 500
        except KeyError as err:
            print(f"Error {err}. Retries left: {max_retries - attempt - 1}")
            prompt = (
                f"{prompt}\n. You got the following error: {err}."
                f"Please ensure your response adheres to the schema"
            )
            attempt += 1
        except ValueError as err:
            print(f"Error {err}. Retries left: {max_retries - attempt - 1}")
            prompt = (
                f"{prompt}\n. You got the following error: {err}."
                f"Please ensure your response adheres to the schema"
            )
    if attempt == max_retries:
        raise llm.InvalidResponseError("Failed to parse response")

    return response  # type: ignore
