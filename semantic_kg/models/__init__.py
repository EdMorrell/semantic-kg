from semantic_kg.models.base import BaseTextGeneration
from semantic_kg.models.llm import OpenAITextGeneration


MODEL_MAP = {"openai": OpenAITextGeneration}


def load_model(model_type: str, **kwargs) -> BaseTextGeneration:
    model = MODEL_MAP[model_type]

    return model(**kwargs)
