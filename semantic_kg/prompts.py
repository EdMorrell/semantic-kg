triplet_prompt_template = """You are going to be provided with a set of triplets describing the relations between different biological entities.

Your goal is to convert this list into a natural language statement that is written like an extract from a scientific paper.

{dataset_rules}

If the relation is described as "parent-child" this refers to the fact that the "source_node" is a sub-type of the "target_node", it does not refer to hereditary relations.

You may use synonyms to descibe the terms if these produce a more natural sounding statement and the statement does not have to follow the same order as the triplets. You are also welcome to add additional contextual information, for example, definitons of the terms involved, however the additional information must not contradict the relations described by the triplets.

Triplets: {triplets}
"""
