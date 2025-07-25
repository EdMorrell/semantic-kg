import os
from pathlib import Path
from typing import Optional

import openai
import google.auth
import google.auth.transport.requests
from google.auth.exceptions import DefaultCredentialsError
from openai import AzureOpenAI, OpenAI

from semantic_kg.models.base import BaseTextGeneration
from semantic_kg.utils import get_hishel_http_client


def get_gcloud_auth_token():
    """Generates auth token for gcloud. If the credentials are expired, refresh them."""
    creds, project = google.auth.default()

    # creds.valid is False, and creds.token is None
    # # Need to refresh credentials to populate those
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    return creds


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


def create_openai_client(cache_dir: Optional[Path] = None):
    """Sets up the OpenAI client"""
    # Creates a hishel object to cache responses
    if cache_dir:
        client = get_hishel_http_client(cache_dir)
    else:
        client = None

    return OpenAI(
        api_key=get_gcloud_auth_token().token,
        base_url=os.environ["OPENAI_API_BASE"],
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


class GeminiTextGeneration(BaseTextGeneration):
    """Model class for Google Gemini text generation models"""

    def __init__(
        self,
        model_id: str,
        cache_dir: Optional[Path] = None,
        temperature: float = 1.0,
        system_prompt: Optional[str] = None,
        structured_output: bool = False,
    ) -> None:
        """Model class for Google Gemini text generation models

        Parameters
        ----------
        model_id : str
            Model ID of the Gemini model
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
        self.cache_dir = cache_dir

        self.client = create_openai_client(cache_dir)

    def _get_response(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        n_responses: int,
        **kwargs,
    ) -> dict:
        return self.client.chat.completions.create(
            model=self.model_id,
            temperature=1,
            max_tokens=max_tokens,
            n=n_responses,
            messages=messages,  # type: ignore
            **kwargs,
        )

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

        try:
            # Attempt to get a response from the client
            response = self._get_response(
                messages=messages,
                max_tokens=max_tokens,
                n_responses=n_responses,
                **kwargs,
            )
        except DefaultCredentialsError or openai.error.AuthenticationError:
            # Refresh the client if credentials are expired
            self.client = create_openai_client(cache_dir=self.cache_dir)
            response = self._get_response(
                messages=messages,
                max_tokens=max_tokens,
                n_responses=n_responses,
                **kwargs,
            )

        response = self.client.chat.completions.create(
            model=self.model_id,
            temperature=1,
            max_tokens=max_tokens,
            n=n_responses,
            messages=messages,  # type: ignore
            **kwargs,
        )
        return [choice.message.content for choice in response.choices]


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    model = GeminiTextGeneration(
        model_id="google/gemini-1.5-flash",
        structured_output=True,
    )
    response = model.generate(
        prompt="What is capital of France?",
        n_responses=1,
        max_tokens=500,
        seed=2342,
    )
    print(response)
