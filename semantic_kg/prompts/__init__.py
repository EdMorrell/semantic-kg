from pydantic import BaseModel

from semantic_kg.prompts import oregano
from semantic_kg.prompts import prime_kg
from semantic_kg.prompts import findkg
from semantic_kg.prompts import utils


class PromptConfig(BaseModel):
    response_model_prompt: str
    entity_extractor_scorer_prompt: str
    kg_extractor_scorer_prompt: str


OREGANO_PROMPT_CONFIG = PromptConfig(
    response_model_prompt=utils.get_default_prompt_template(
        fewshot_examples=oregano.oregano_fewshot_examples,
        prompt_rules=oregano.oregano_prompt_rules,
    ),
    entity_extractor_scorer_prompt=utils.get_entity_extractor_system_prompt(
        node_types=oregano.NODE_TYPES,
        entity_extractor_fewshot_examples=oregano.entity_extractor_fewshot_examples,
    ),
    kg_extractor_scorer_prompt=utils.get_kg_extractor_system_prompt(
        edge_types=oregano.EDGE_TYPES,
        fewshot_examples=oregano.kg_extractor_fewshot_example,
        directed=True,
        incl_valid_directions=True,
        valid_edge_directions=oregano.VALID_EDGE_DIRECTIONS,
    ),
)


PRIME_KG_PROMPT_CONFIG = PromptConfig(
    response_model_prompt=utils.get_default_prompt_template(
        fewshot_examples=prime_kg.prime_kg_fewshot_examples,
        prompt_rules=prime_kg.prime_kg_prompt_rules,
        directed=True,
    ),
    entity_extractor_scorer_prompt=utils.get_entity_extractor_system_prompt(
        node_types=prime_kg.NODE_TYPES,
        entity_extractor_fewshot_examples=prime_kg.entity_extractor_fewshot_examples,
    ),
    kg_extractor_scorer_prompt=utils.get_kg_extractor_system_prompt(
        edge_types=prime_kg.EDGE_TYPES,
        fewshot_examples=prime_kg.kg_extractor_fewshot_examples,
        incl_valid_directions=False,
    ),
)


FINDKG_PROMPT_CONFIG = PromptConfig(
    response_model_prompt=utils.get_default_prompt_template(
        fewshot_examples=findkg.findkg_fewshot_examples,
        prompt_rules=findkg.findkg_prompt_rules,
    ),
    entity_extractor_scorer_prompt=utils.get_entity_extractor_system_prompt(
        node_types=findkg.NODE_TYPES,
        entity_extractor_fewshot_examples=findkg.entity_extractor_fewshot_examples,
    ),
    kg_extractor_scorer_prompt=utils.get_kg_extractor_system_prompt(
        edge_types=findkg.EDGE_TYPES,
        fewshot_examples=findkg.kg_extractor_fewshot_examples,
        incl_valid_directions=False,
    ),
)
