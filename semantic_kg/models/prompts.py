triple_prompt_template = """You are going to be provided with a set of triples describing the relations between different biological entities.

Your goal is to convert this list into a natural language statement that is written like an extract from a scientific paper.

{dataset_rules}

You may use synonyms to descibe the terms if these produce a more natural sounding statement and the statement does not have to follow the same order as the triples. You are also welcome to add additional contextual information, for example, definitons of the terms involved, however the additional information must not contradict the relations described by the triples.

triples: {triples}
"""


prime_kg_prompt_rules = """If the relation is described as "parent-child" this refers to the fact that the "source_node" is a sub-type of the "target_node", it does not refer to hereditary relations.

In your final response, do NOT put name of any node or relation in quotes.
For example for the node `{{"name": "Facial palsy"}}`:
    '...demonstrates several phenotypic manifestations, such as "Facial palsy"...' is **not** allowed
    '...demonstrates several phenotypic manifestations, such as 'Facial palsy'...' is **not** allowed
    '...demonstrates several phenotypic manifestations, such as facial palsy...' is allowed
"""
