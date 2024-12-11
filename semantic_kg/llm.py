import os

from openai import AzureOpenAI



def create_azure_openai_client():
    """Sets up the Azure OpenAI client"""
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )


class OpenAITextGeneration:
    def __init__(self, model_id: str, temperature: float = 1.0) -> None:
        """Model class for Azure OpenAI text generation models

        Parameters
        ----------
        model_id : str
            Deployment ID of model in Azure
        temperature : float, optional
            Temperature for generation, by default 1.0
        """
        super().__init__()
        self.model_id = model_id
        self.temperature = temperature

        self.client = create_azure_openai_client()

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