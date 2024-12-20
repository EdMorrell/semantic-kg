from functools import partial

from semantic_kg.prompts import scorer
from semantic_kg.prompts import schema
from semantic_kg.prompts.default import (
    triple_prompt_template,
    triple_prompt_template_directed,
)

NODE_TYPES = "['gene/protein', 'drug', 'effect/phenotype', 'disease', 'biological_process', 'molecular_function', 'cellular_component', 'exposure', 'pathway', 'anatomy']"
EDGE_TYPES = "['ppi', 'carrier', 'enzyme', 'target', 'transporter', 'contraindication', 'indication', 'off-label use', 'synergistic interaction', 'associated with', 'parent-child', 'phenotype absent', 'phenotype present', 'side effect', 'interacts with', 'linked to', 'expression present', 'expression absent']"


prime_kg_fewshot_examples = """
triple = [
    {{"source_node": {{"name": "Typhoid Vaccine Live"}}, "relation": {{"name": "synergistic interaction"}}, "target_node": {{"name": "Ruzasvir"}}}},
    {{"source_node": {{"name": "Bacillus calmette-guerin substrain tice live antigen"}}, "relation": {{"name": "synergistic interaction"}}, "target_node": {{"name": "Ruzasvir"}}}},
]

You could say:
"Typhoid Vaccine Live has a synergistic interaction with Ruzasvir additionally the Bacillus calmette-guerin substrain tice live antigen has a synergistic interaction with Ruzasvir"

or given the triple:

triple = [
    {{'source_node': {{'name': 'EGR2'}}, 'relation': {{'name': 'interacts with'}}, 'target_node': {{'name': 'muscle tissue disease'}}}},
    {{'source_node': {{'name': 'muscle cancer'}}, 'relation': {{'name': 'parent-child'}}, 'target_node': {{'name': 'muscle tissue disease'}}}},
    {{'source_node': {{'name': 'KCNQ1'}}, 'relation': {{'name': 'interacts with'}}, 'target_node': {{'name': 'muscle tissue disease'}}}},
]

You could say:
"Several genes including ERG2 and KCNQ1 interact with muscle tissue disease, a sub-type of muscle cancer"


or given the triple:

triple = [
    {{'source_node': {{'name': 'spastic paraplegia-severe developmental delay-epilepsy syndrome'}}, 'relation': {{'name': 'phenotype present'}}, 'target_node': {{'name': 'Focal myoclonic seizure'}}}},
    {{'source_node': {{'name': 'Craniosynostosis'}}, 'relation': {{'name': 'phenotype present'}}, 'target_node': {{'name': 'skeletal dysplasia-T-cell immunodeficiency-developmental delay syndrome'}}}},
    {{'source_node': {{'name': 'Focal myoclonic seizure'}}, 'relation': {{'name': 'phenotype present'}}, 'target_node': {{'name': 'progressive myoclonic epilepsy'}}}},
    {{'source_node': {{'name': 'Focal myoclonic seizure'}}, 'relation': {{'name': 'phenotype present'}}, 'target_node': {{'name': 'mitochondrial complex II deficiency, nuclear'}}}},
    {{'source_node': {{'name': 'Focal myoclonic seizure'}}, 'relation': {{'name': 'phenotype present'}}, 'target_node': {{'name': 'skeletal dysplasia-T-cell immunodeficiency-developmental delay syndrome'}}}},
    {{'source_node': {{'name': 'Frontal bossing'}}, 'relation': {{'name': 'phenotype present'}}, 'target_node': {{'name': 'skeletal dysplasia-T-cell immunodeficiency-developmental delay syndrome'}}}}
]

You could say:
"Focal myoclonic seizures are a common phenotype present in a range of conditions, including spastic paraplegia-severe developmental delay-epilepsy syndrome, progressive myoclonic epilepsy, nuclear mitochondrial complex II deficiency and skeletal dysplasia-T-cell immunodeficiency-developmental delay syndrome. Other phenotypes are also observed in skeletal dysplasia-T-cell immunodeficiency-developmental delay syndrome including craniosynostosis and frontal bossing"
"""


prime_kg_prompt_rules = """If the relation is described as "parent-child" this refers to the fact that the "target_node" is a sub-type of the "parent_node", it does not refer to hereditary relations.

In your final response, do NOT put name of any node or relation in quotes.
For example for the node `{{"name": "Facial palsy"}}`:
    '...demonstrates several phenotypic manifestations, such as "Facial palsy"...' is **not** allowed
    '...demonstrates several phenotypic manifestations, such as 'Facial palsy'...' is **not** allowed
    '...demonstrates several phenotypic manifestations, such as facial palsy...' is allowed
"""


entity_extractor_fewshot_examples = """Example input: "Asymmetry of the position of the ears is a specific sub-category of the broader concept of asymmetry of the ears. This phenotypic variation has been observed to be present in patients diagnosed with developmental and epileptic encephalopathy, and in those affected by 8q24.3 microdeletion syndrome."

Expected response: {{
    "entities": [
        "asymmetry of the position of the ears",
        "asymmetry of the ears",
        "developmental and epileptic encephalopathy",
        "8q24.3 microdeletion syndrome",
    ]
}}
"""

entity_extractor_with_type_fewshot_examples = """Example input: "Asymmetry of the position of the ears is a specific sub-category of the broader concept of asymmetry of the ears. This phenotypic variation has been observed to be present in patients diagnosed with developmental and epileptic encephalopathy, and in those affected by 8q24.3 microdeletion syndrome."

Expected response: {{
    "entities": [
        {{"name": "asymmetry of the position of the ears", "node_type": "effect/phenotype"}},
        {{"name": "asymmetry of the ears", "node_type": "effect/phenotype"}},
        {{"name": "developmental and epileptic encephalopathy", "node_type": "disease"}},
        {{"name": "8q24.3 microdeletion syndrome", "node_type": "disease}},
    ]
}}
"""

kg_extractor_fewshot_examples = """
1. Example entities: ["asymmetry of the ears", "aysmmetry of the position of the ears", "developmental and epileptic encephalopathy", "8q24.3 microdeletion syndrome"]
1. Example text: "Asymmetry of the position of the ears is a specific sub-category of the broader concept of asymmetry of the ears. This phenotypic variation has been observed to be present in patients diagnosed with developmental and epileptic encephalopathy, and in those affected by 8q24.3 microdeletion syndrome."

1. Expected output:
{{"triples": [{{"source_node": {{"name": "Asymmetry of the ears"}}, "relation": {{"name": "parent-child"}}, "target_node": {{"name": "Asymmetry of the position of the ears"}}}}, {{"source_node": {{"name": "Asymmetry of the position of the ears"}}, "relation": {{"name": "phenotype-present"}}, "target_node": {{"name": "developmental and epileptic encephalopathy"}}}}, {{"source_node": {{"name": "Asymmetry of the position of the ears"}}, "relation": {{"name": "phenotype-present"}}, "target_node": {{"name": "8q24.3 microdeletion syndrome"}}}}]}}


2. Example entities: ["spastic paraplegia-severe developmental delay-epilepsy syndrome", "focal myoclonic seizure", "craniosynostosis", "skeletal dysplasia-T-cell immunodeficiency-developmental delay syndrome", "skeletal dysplasia-T-cell immunodeficiency-developmental delay syndrome", "progressive myoclonic epilepsy", "mitochondrial complex II deficiency, nuclear", "frontal bossing"]
2. Example text: "Focal myoclonic seizures are a common phenotype present in a range of conditions, including spastic paraplegia-severe developmental delay-epilepsy syndrome, progressive myoclonic epilepsy, nuclear mitochondrial complex II deficiency and skeletal dysplasia-T-cell immunodeficiency-developmental delay syndrome. Other phenotypes are also observed in skeletal dysplasia-T-cell immunodeficiency-developmental delay syndrome including craniosynostosis and frontal bossing"

2. Expected output:
{{
    "triples": [
            {{'source_node': {{'name': 'spastic paraplegia-severe developmental delay-epilepsy syndrome'}}, 'relation': {{'name': 'phenotype present'}}, 'target_node': {{'name': 'Focal myoclonic seizure'}}}},
            {{'source_node': {{'name': 'Craniosynostosis'}}, 'relation': {{'name': 'phenotype present'}}, 'target_node': {{'name': 'skeletal dysplasia-T-cell immunodeficiency-developmental delay syndrome'}}}},
            {{'source_node': {{'name': 'Focal myoclonic seizure'}}, 'relation': {{'name': 'phenotype present'}}, 'target_node': {{'name': 'progressive myoclonic epilepsy'}}}},
            {{'source_node': {{'name': 'Focal myoclonic seizure'}}, 'relation': {{'name': 'phenotype present'}}, 'target_node': {{'name': 'mitochondrial complex II deficiency, nuclear'}}}},
            {{'source_node': {{'name': 'Focal myoclonic seizure'}}, 'relation': {{'name': 'phenotype present'}}, 'target_node': {{'name': 'skeletal dysplasia-T-cell immunodeficiency-developmental delay syndrome'}}}},
            {{'source_node': {{'name': 'Frontal bossing'}}, 'relation': {{'name': 'phenotype present'}}, 'target_node': {{'name': 'skeletal dysplasia-T-cell immunodeficiency-developmental delay syndrome'}}}}
    ]

}}

3. Example entities: ["ERG2", "KCNQ1", "muscle tissue disease", "muscle cancer"]
3. Example text: "Several genes including ERG2 and KCNQ1 interact with muscle tissue disease, a sub-type of muscle cancer"

3. Expected output:
{{
    "triples": [
            {{'source_node': {{'name': 'EGR2'}}, 'relation': {{'name': 'interacts with'}}, 'target_node': {{'name': 'muscle tissue disease'}}}},
            {{'source_node': {{'name': 'muscle cancer'}}, 'relation': {{'name': 'parent-child'}}, 'target_node': {{'name': 'muscle tissue disease'}}}},
            {{'source_node': {{'name': 'KCNQ1'}}, 'relation': {{'name': 'interacts with'}}, 'target_node': {{'name': 'muscle tissue disease'}}}},
    ]
}}
"""


kg_extractor_single_step_fewshot_examples = """
1. Example input: "Asymmetry of the position of the ears is a specific sub-category of the broader concept of asymmetry of the ears. This phenotypic variation has been observed to be present in patients diagnosed with developmental and epileptic encephalopathy, and in those affected by 8q24.3 microdeletion syndrome."

1. Expected output:
{{"triples": [{{"source_node": {{"name": "Asymmetry of the ears"}}, "relation": {{"name": "parent-child"}}, "target_node": {{"name": "Asymmetry of the position of the ears"}}}}, {{"source_node": {{"name": "Asymmetry of the position of the ears"}}, "relation": {{"name": "phenotype-present"}}, "target_node": {{"name": "developmental and epileptic encephalopathy"}}}}, {{"source_node": {{"name": "Asymmetry of the position of the ears"}}, "relation": {{"name": "phenotype-present"}}, "target_node": {{"name": "8q24.3 microdeletion syndrome"}}}}]}}


2. Example input: "Focal myoclonic seizures are a common phenotype present in a range of conditions, including spastic paraplegia-severe developmental delay-epilepsy syndrome, progressive myoclonic epilepsy, nuclear mitochondrial complex II deficiency and skeletal dysplasia-T-cell immunodeficiency-developmental delay syndrome. Other phenotypes are also observed in skeletal dysplasia-T-cell immunodeficiency-developmental delay syndrome including craniosynostosis and frontal bossing"

2. Expected output:
{{
    "triples": [
            {{'source_node': {{'name': 'spastic paraplegia-severe developmental delay-epilepsy syndrome'}}, 'relation': {{'name': 'phenotype present'}}, 'target_node': {{'name': 'Focal myoclonic seizure'}}}},
            {{'source_node': {{'name': 'Craniosynostosis'}}, 'relation': {{'name': 'phenotype present'}}, 'target_node': {{'name': 'skeletal dysplasia-T-cell immunodeficiency-developmental delay syndrome'}}}},
            {{'source_node': {{'name': 'Focal myoclonic seizure'}}, 'relation': {{'name': 'phenotype present'}}, 'target_node': {{'name': 'progressive myoclonic epilepsy'}}}},
            {{'source_node': {{'name': 'Focal myoclonic seizure'}}, 'relation': {{'name': 'phenotype present'}}, 'target_node': {{'name': 'mitochondrial complex II deficiency, nuclear'}}}},
            {{'source_node': {{'name': 'Focal myoclonic seizure'}}, 'relation': {{'name': 'phenotype present'}}, 'target_node': {{'name': 'skeletal dysplasia-T-cell immunodeficiency-developmental delay syndrome'}}}},
            {{'source_node': {{'name': 'Frontal bossing'}}, 'relation': {{'name': 'phenotype present'}}, 'target_node': {{'name': 'skeletal dysplasia-T-cell immunodeficiency-developmental delay syndrome'}}}}
    ]

}}

3. Example input: "Several genes including ERG2 and KCNQ1 interact with muscle tissue disease, a sub-type of muscle cancer"
3. Expected output:
{{
    "triples": [
            {{'source_node': {{'name': 'EGR2'}}, 'relation': {{'name': 'interacts with'}}, 'target_node': {{'name': 'muscle tissue disease'}}}},
            {{'source_node': {{'name': 'muscle cancer'}}, 'relation': {{'name': 'parent-child'}}, 'target_node': {{'name': 'muscle tissue disease'}}}},
            {{'source_node': {{'name': 'KCNQ1'}}, 'relation': {{'name': 'interacts with'}}, 'target_node': {{'name': 'muscle tissue disease'}}}},
    ]
}}
"""


def get_default_prompt_template(directed: bool = True) -> str:
    """Helper function to generate a default prompt template for PrimeKG"""
    template = triple_prompt_template_directed if directed else triple_prompt_template
    rule_formatter = partial(
        template.format,
        fewshot_examples=prime_kg_fewshot_examples,
        dataset_rules=prime_kg_prompt_rules,
    )

    return rule_formatter(triples="{triples}")


def get_entity_extractor_system_prompt() -> str:
    """Helper function to generate the entity-extractor system prompt"""
    # TODO: Support generating a system prompt with type
    return scorer.entity_extractor_system_prompt_template.format(
        node_types=NODE_TYPES,
        response_schema=schema.entity_response_schema,
        fewshot_examples=entity_extractor_fewshot_examples,
    )


def get_kg_extractor_system_prompt(directed: bool = True) -> str:
    """Helper function to generate the KG-extractor system prompt"""
    template = (
        scorer.kg_extractor_system_prompt_template_directed
        if directed
        else scorer.kg_extractor_system_prompt_template
    )

    return template.format(
        edge_types=EDGE_TYPES,
        response_schema=schema.triple_response_format,
        fewshot_examples=kg_extractor_fewshot_examples,
    )
