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


prime_kg_prompt_rules = """If the relation is described as "parent-child" this refers to the fact that the "source_node" is a sub-type of the "target_node", it does not refer to hereditary relations.

In your final response, do NOT put name of any node or relation in quotes.
For example for the node `{{"name": "Facial palsy"}}`:
    '...demonstrates several phenotypic manifestations, such as "Facial palsy"...' is **not** allowed
    '...demonstrates several phenotypic manifestations, such as 'Facial palsy'...' is **not** allowed
    '...demonstrates several phenotypic manifestations, such as facial palsy...' is allowed
"""


prime_kg_scorer_system_prompt_template = """You will be provided with a statement describing the relationship between various biological entities.

The different entities described are of the following types: {node_types}.

Entities are related to the other entities via the following relationships: {edge_types}

Please extract a list of triples from the provided statement describing the identity and relationship of different entities.

Please provide your response in valid JSON using the following response schema: {response_schema}

Examples:

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
