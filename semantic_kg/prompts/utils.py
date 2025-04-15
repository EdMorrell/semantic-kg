from typing import Optional
from functools import partial


from semantic_kg.prompts import schema
from semantic_kg.prompts.default import (
    triple_prompt_template,
    triple_prompt_template_directed,
)
from semantic_kg.prompts import scorer


def get_default_prompt_template(
    fewshot_examples: str, prompt_rules: Optional[str] = None, directed: bool = True
) -> str:
    """Generates a dataset-specific prompt template from the default template

    Parameters
    ----------
    fewshot_examples : str
        Few-shot examples for the given prompt
    prompt_rules : Optional[str], optional
        Dataset-specific rules for KG generation, by default None
    directed : bool, optional
        Whether or not to use a directed prompt template, by default True

    Returns
    -------
    str
        Prompt template
    """
    template = triple_prompt_template_directed if directed else triple_prompt_template

    if not prompt_rules:
        prompt_rules = ""

    rule_formatter = partial(
        template.format,
        fewshot_examples=fewshot_examples,
        dataset_rules=prompt_rules,
    )

    return rule_formatter(triples="{triples}")


def get_entity_extractor_system_prompt(
    node_types: str, entity_extractor_fewshot_examples: str
) -> str:
    """Helper function to generate the entity-extractor system prompt"""
    # TODO: Support generating a system prompt with type
    return scorer.entity_extractor_system_prompt_template.format(
        node_types=node_types,
        response_schema=schema.entity_response_schema,
        fewshot_examples=entity_extractor_fewshot_examples,
    )


def get_kg_extractor_system_prompt(
    edge_types: str,
    fewshot_examples: str,
    directed: bool = True,
    incl_valid_directions: bool = False,
    valid_edge_directions: Optional[str] = None,
) -> str:
    """Generates a dataset-specific prompt for the Knowledge-Graph extractor

    Parameters
    ----------
    edge_types : str
        A string indicating all valid types of edge
    fewshot_examples : str
        Few-shot examples for the extraction
    directed : bool, optional
        Whether or not we expect the scorer to return a directed graph, by default True
    incl_valid_directions : bool, optional
        If True then edges are only valid between certain edge-types (defined by
        `valid_edge_directions`), by default False
    valid_edge_directions : Optional[str], optional
       A string representing node-types for which an edge is valid. For example:
       ```
       valid_edge_directions = "'GENE -> DISEASE', 'DISEASE -> PHENOTYPE', 'COMPOUND -> GENE'"
       ```
       Must be specified if `incl_valid_directions` is True. By default None.

    Returns
    -------
    str
       Prompt template
    """
    if incl_valid_directions and not valid_edge_directions:
        raise ValueError(
            "`valid_edge_directions` can't be `None` if `incl_valid_directions` is True"
        )

    if directed and incl_valid_directions:
        template = scorer.kg_extractor_system_prompt_template_directed_valid_directions
    elif directed and not incl_valid_directions:
        template = scorer.kg_extractor_system_prompt_template_directed
    else:
        template = scorer.kg_extractor_system_prompt_template

    kwargs = {}
    if incl_valid_directions:
        kwargs["edge_directions"] = valid_edge_directions

    return template.format(
        edge_types=edge_types,
        response_schema=schema.triple_response_format,
        fewshot_examples=fewshot_examples,
        **kwargs,
    )
