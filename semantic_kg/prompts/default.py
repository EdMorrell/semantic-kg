old_triple_prompt_template = """You are going to be provided with a set of triples describing the relations between different biological entities.

Your goal is to convert this list into a natural language statement that is written like an extract from a scientific paper.

{dataset_rules}

You may use synonyms to descibe the terms if these produce a more natural sounding statement and the statement does not have to follow the same order as the triples. You are also welcome to add additional contextual information, for example, definitons of the terms involved, however the additional information must not contradict the relations described by the triples.

triples: {triples}
"""

triple_prompt_template = """You are going to be given a list of triples from a knowledge graph. Each triple consists of a subject, a relation, and an object.

Your goal is to express this triple in a continuous natural language statement suitable for a general or a scientific audience.

For example, given the triple:

{fewshot_examples}

Rules:

You must use all of the entities provided in the triple and please include each node verbatim but do not use quotes.

Do NOT list the items in the triple as a list. Instead, write a sentence or paragraph that describes the relationship between every item in the triple.

You can also add additional information to the triple to make the relationship more clear, however you must include all the triples in your response.

{dataset_rules}

Triples: {triples}
"""


triple_response_format = {
    "type": "object",
    "properties": {
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_node": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                        "additionalProperties": False,
                    },
                    "relation": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                        "additionalProperties": False,
                    },
                    "target_node": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                        "additionalProperties": False,
                    },
                },
                "required": ["source_node", "relation", "target_node"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["triples"],
    "additionalProperties": False,
    "strict": True,
}
