from typing import Optional
from semantic_kg.models.base import BaseTextGeneration
from semantic_kg.models.utils import structured_generation_helper
from semantic_kg.utils import find_field_placeholders


class KGReconstuctionModel(BaseTextGeneration):
    def __init__(
        self,
        entity_generation_model: BaseTextGeneration,
        kg_generation_model: BaseTextGeneration,
        entity_generation_user_prompt_template: Optional[str] = None,
        kg_generation_user_prompt_template: Optional[str] = None,
        max_retries: int = 4,
        increase_tokens_on_retry: bool = True,
    ) -> None:
        """Model pipeline for reconstructing a knowledge graph from a statement

        Class chains together 2 LLM calls into the following steps:
        1. Generate entities from a statement
        2. Generate a knowledge graph from the entities and statement

        NOTE: This class inherits from `BaseTextGeneration` and is intended to be
        used in the same way as a single model

        Parameters
        ----------
        entity_generation_model : BaseTextGeneration
            Model to generate entities from a statement
        kg_generation_model : BaseTextGeneration
            Model to generate a knowledge graph from entities and statement
        entity_generation_user_prompt_template : Optional[str], optional
            User prompt template for entity generation. Must contain a placholder
            field called `statement`. If none provided then prompt is passed to
            model as is, by default None
        kg_generation_user_prompt_template : Optional[str], optional
            User prompt template for KG generation. Must contain placeholder fields
            called `entities` and `text`. If none provided then passes entities and
            statement to the model as is, by default None
        """
        self.entity_generation_model = entity_generation_model
        self.kg_generation_model = kg_generation_model

        self.entity_generation_user_prompt = entity_generation_user_prompt_template
        if self.entity_generation_user_prompt:
            if "statement" not in find_field_placeholders(
                self.entity_generation_user_prompt
            ):
                raise ValueError(
                    "`entity_generation_user_prompt` must contain a "
                    "placeholder field called `statement`"
                )

        if kg_generation_user_prompt_template:
            field_placeholders = find_field_placeholders(
                kg_generation_user_prompt_template
            )
            if "entities" not in field_placeholders or "text" not in field_placeholders:
                raise ValueError(
                    "`kg_generation_user_prompt` must contain a "
                    "placeholder field called `entities`"
                )
            self.kg_generation_user_prompt = kg_generation_user_prompt_template
        else:
            self.kg_generation_user_prompt = """Entities: {entities}\n\nText: {text}"""

        self.max_retries = max_retries
        self.increase_tokens_on_retry = increase_tokens_on_retry

    def generate(
        self, prompt: str, n_responses: int, max_tokens: int, seed: int | None = None
    ) -> list[str | None]:
        """Method to generate responses"""
        if self.entity_generation_user_prompt:
            prompt = self.entity_generation_user_prompt.format(statement=prompt)

        entity_response = structured_generation_helper(
            self.entity_generation_model,
            prompt,
            max_retries=self.max_retries,
            max_tokens=max_tokens,
            seed=seed,
            increase_tokens_on_retry=self.increase_tokens_on_retry,
        )
        entities = entity_response["entities"]

        prompt = self.kg_generation_user_prompt.format(entities=entities, text=prompt)

        kg_generation_response = structured_generation_helper(
            self.kg_generation_model,
            prompt,
            max_retries=self.max_retries,
            max_tokens=max_tokens,
            seed=seed,
            increase_tokens_on_retry=self.increase_tokens_on_retry,
        )

        # Format response into expected format for `BaseTextGeneration`
        final_response = [str(kg_generation_response)]

        return final_response  # type: ignore
