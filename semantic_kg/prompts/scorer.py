entity_extractor_system_prompt_template = """You will be provided with a statement describing the relationship between various entities.

The different entities described are of the following types: {node_types}.

Please extract a list of all the entities of the types described above from the given passage.

Please provide your response in valid JSON using the following response schema: {response_schema}

For example:

{fewshot_examples}

"""


kg_extractor_system_prompt_template = """You will be provided with a text describing the relationship between various entities

You will also be provided with a list of entities contained within that statement.

Entities are related to the other entities via the following relationships: {edge_types}

Your goal is to extract the relationships between the entities using the provided text and generate a list of triples in valid JSON using the following schema: {response_schema}

Examples:

{fewshot_examples}

"""


kg_extractor_system_prompt_template_directed = """You will be provided with a text describing the directed relationships between various entities

You will also be provided with a list of entities contained within that statement.

Entities are related to the other entities via the following directed relationships: {edge_types}

Your goal is to extract the relationships between the entities as a directed graph and represent that graph as a list of triples in valid json using the following schema: {response_schema}

The triples should be represented in directed order with the relation direction going from "source_node" to "target_node"

Examples:

{fewshot_examples}

"""

kg_extractor_system_prompt_template_directed_valid_directions = """You will be provided with a text describing the directed relationships between various entities

You will also be provided with a list of entities contained within that statement.

Entities are related to the other entities via the following directed relationships: {edge_types}

Your goal is to extract the relationships between the entities as a directed graph and represent that graph as a list of triples in valid json using the following schema: {response_schema}

The triples should be represented in directed order with the relation direction going from "source_node" to "target_node"

The valid edge directions are {edge_directions}

Examples:

{fewshot_examples}

"""


# NOTE: Not recommended to use as 2-step method appeared better in practice
single_step_kg_extractor_system_prompt_template = """You will be provided with a statement describing the relationship between various entities.

The different entities described are of the following types: {node_types}.

Entities are related to the other entities via the following relationships: {edge_types}

Please extract a list of triples from the provided statement describing the identity and relationship of different entities.

Please provide your response in valid JSON using the following response schema: {response_schema}

Examples:

{fewshot_examples}

"""
