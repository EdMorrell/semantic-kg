from semantic_kg.models.base import BaseTextGeneration
from semantic_kg.models.llm import OpenAITextGeneration


MODEL_MAP = {"openai": OpenAITextGeneration}


def load_model(model_name: str, **kwargs) -> BaseTextGeneration:
    model = MODEL_MAP[model_name]

    return model(**kwargs)
