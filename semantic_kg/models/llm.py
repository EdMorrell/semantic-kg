import os
from pathlib import Path
from typing import Optional

from openai import AzureOpenAI

from semantic_kg.models.base import BaseTextGeneration
from semantic_kg.utils import get_hishel_http_client


class InvalidResponseError(Exception):
    """Error class for invalid LLM response"""

    pass


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
        self,
        model_id: str,
        cache_dir: Optional[Path] = None,
        temperature: float = 1.0,
        system_prompt: Optional[str] = None,
        structured_output: bool = False,
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
        system_prompt : str, optional
            System prompt to use. If not provided then only user prompt used.
        structured_output : bool, optional
            Whether to provide a JSON structured output, defaults to False.
        """
        super().__init__()
        self.model_id = model_id
        self.temperature = temperature
        if not cache_dir:
            print("No cache directory provided, responses will not be cached")
        self.system_prompt = system_prompt
        self.structured_output = structured_output

        self.client = create_azure_openai_client(cache_dir)

    def generate(
        self,
        prompt: str,
        n_responses: int,
        max_tokens: int,
        seed: Optional[int] = None,
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
        seed : Optional[int], optional
            Random seed for generation

        Returns
        -------
        list[str | None]
            List of generated responses
        """
        if self.system_prompt:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        kwargs = {}
        if self.structured_output:
            kwargs["response_format"] = {"type": "json_object"}

        if seed:
            kwargs["seed"] = seed

        response = self.client.chat.completions.create(
            model=self.model_id,
            temperature=1,
            max_tokens=max_tokens,
            n=n_responses,
            messages=messages,  # type: ignore
            **kwargs,
        )
        return [choice.message.content for choice in response.choices]
