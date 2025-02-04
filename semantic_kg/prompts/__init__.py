from pydantic import BaseModel

from semantic_kg.prompts import oregano
from semantic_kg.prompts import prime_kg


class PromptConfig(BaseModel):
    response_model_prompt: str
    entity_extractor_scorer_prompt: str
    kg_extractor_scorer_prompt: str


OREGANO_PROMPT_CONFIG = PromptConfig(
    response_model_prompt=oregano.get_default_prompt_template(),
    entity_extractor_scorer_prompt=oregano.get_entity_extractor_system_prompt(),
    kg_extractor_scorer_prompt=oregano.get_kg_extractor_system_prompt(),
)


PRIME_KG_PROMPT_CONFIG = PromptConfig(
    response_model_prompt=prime_kg.get_default_prompt_template(),
    entity_extractor_scorer_prompt=prime_kg.get_entity_extractor_system_prompt(),
    kg_extractor_scorer_prompt=prime_kg.get_kg_extractor_system_prompt(),
)
