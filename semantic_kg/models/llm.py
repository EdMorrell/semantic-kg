import os
from pathlib import Path
from typing import Optional

from openai import AzureOpenAI

from semantic_kg.models.base import BaseTextGeneration
from semantic_kg.models.utils import get_hishel_http_client


def create_azure_openai_client(cache_dir: Optional[Path] = None):
    """Sets up the Azure OpenAI client"""
    # Creates a hishel object to cache responses
    if cache_dir:
        client = get_hishel_http_client(cache_dir)
    else:
        client = None

    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        http_client=client,
    )


class OpenAITextGeneration(BaseTextGeneration):
    def __init__(
        self, model_id: str, cache_dir: Optional[Path] = None, temperature: float = 1.0
    ) -> None:
        """Model class for Azure OpenAI text generation models

        Parameters
        ----------
        model_id : str
            Deployment ID of model in Azure
        cache_dir : Optional[Path], optional
            Path to use for caching responses. If not provided then responses not cached
        temperature : float, optional
            Temperature for generation, by default 1.0
        """
        super().__init__()
        self.model_id = model_id
        self.temperature = temperature
        if not cache_dir:
            print("No cache directory provided, responses will not be cached")

        self.client = create_azure_openai_client(cache_dir)

    def generate(
        self, prompt: str, n_responses: int, max_tokens: int
    ) -> list[str | None]:
        """Generates a list of responses to a prompt

        Parameters
        ----------
        prompt : str
            Prompt
        n_responses : int
            Number of responses to generate
        max_tokens : int
            Max number of tokens per response

        Returns
        -------
        list[str | None]
            List of generated responses
        """
        response = self.client.chat.completions.create(
            model=self.model_id,
            temperature=1,
            max_tokens=max_tokens,
            n=n_responses,
            messages=[{"role": "user", "content": prompt}],
        )
        return [choice.message.content for choice in response.choices]
