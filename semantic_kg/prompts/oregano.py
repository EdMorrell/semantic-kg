NODE_TYPES = "['COMPOUND', 'GENE', 'DISEASE', 'PROTEIN', 'MOLECULE', 'ACTIVITY', 'EFFECT', 'PHENOTYPE', 'PATHWAY', 'INDICATION', 'SIDE_EFFECT']"

EDGE_TYPES = "['has_target', 'increase_activity', 'has_activity', 'decrease_activity', 'increase_effect', 'has_effect', 'decrease_effect', 'increase_efficacy', 'decrease_efficacy', 'causes_condition', 'has_phenotype', 'is_affecting', 'is_substance_that_treats', 'acts_within', 'has_indication', 'has_side_effect', 'gene_product_of']"

VALID_EDGE_DIRECTIONS = "['COMPOUND -> PROTEIN', 'COMPOUND -> MOLECULE', 'COMPOUND -> ACTIVITY', 'COMPOUND -> EFFECT', 'COMPOUND -> COMPOUND', 'GENE -> DISEASE', 'DISEASE -> PHENOTYPE', 'COMPOUND -> GENE', 'COMPOUND -> DISEASE', 'GENE -> PATHWAY', 'COMPOUND -> INDICATION', 'COMPOUND -> SIDE', 'PROTEIN -> GENE']"


oregano_fewshot_examples = """
triple = [
    {{'source_node': {{'name': 'GFRA2'}}, 'relation': {{'name': 'acts_within'}}, 'target_node': {{'name': 'NCAM1 interactions'}}}},
    {{'source_node': {{'name': 'GFRA2'}}, 'relation': {{'name': 'acts_within'}}, 'target_node': {{'name': 'RAF/MAP kinase cascade'}}}},
    {{'source_node': {{'name': 'GFRA2'}}, 'relation': {{'name': 'acts_within'}}, 'target_node': {{'name': 'RET signaling'}}}}
]

You could say:
"The gene GRFA2 acts within several pathways including NCAM1 interactions pathway, RAF/MAP kinase cascade and RET signaling.

or given the triple:

triple = [
    {{'source_node': {{'name': 'nephrotic syndrome'}}, 'relation': {{'name': 'has_phenotype'}}, 'target_node': {{'name': 'microcephaly'}}}},
    {{'source_node': {{'name': 'SGPL1'}}, 'relation': {{'name': 'causes_condition'}}, 'target_node': {{'name': 'nephrotic syndrome'}}}},
    {{'source_node': {{'name': 'SGPL1'}}, 'relation': {{'name': 'acts_within'}}, 'target_node': {{'name': 'Sphingolipid de novo biosynthesis'}}}}
]

You could say:
"The gene SGPL1 which acts within the sphingolipid de novo biosynthesis pathway and causes nephrotic syndrome, a disease characterised by symptoms such as microcephaly"


or given the triple:

triple = [
    {{'source_node': {{'name': 'methyl l-phenylalaninate'}}, 'relation': {{'name': 'has_target'}}, 'target_node': {{'name': 'fimbrial protein'}}}},
    {{'source_node': {{'name': 'methyl l-phenylalaninate'}}, 'relation': {{'name': 'has_target'}}, 'target_node': {{'name': 'prothrombin'}}}},
    {{'source_node': {{'name': 'prothrombin'}}, 'relation': {{'name': 'gene_product_of'}}, 'target_node': {{'name': 'F2'}}}},
    {{'source_node': {{'name': 'F2'}}, 'relation': {{'name': 'causes_condition'}}, 'target_node': {{'name': 'pregnancy loss'}}}}
]

You could say:
"The drug methyl l-phenylalaninate targets both the fimbrial protein and prothrombin. Prothrombin is a gene product of F2 which is known to cause pregnancy loss.
"""

oregano_prompt_rules = """In your final response, do NOT put name of any node or relation in quotes.
For example for the node `{{"name": "2-(6-methylpyridin-2-yl)-n-pyridin-4-ylquinazolin-4-amine"}}`:
    'The drug  "2-(6-methylpyridin-2-yl)-n-pyridin-4-ylquinazolin-4-amine"...' is **not** allowed
    '... is activate by the drug "2-(6-methylpyridin-2-yl)-n-pyridin-4-ylquinazolin-4-amine"' is **not** allowed
    '... is activate by the drug 2-(6-methylpyridin-2-yl)-n-pyridin-4-ylquinazolin-4-amine' is allowed

Ensure you correctly capitalize the entity in your statement. The capitalization does not need to match the provided entity.
For example for the node: `{{"name": "Common Pathway of Fibrin Clot Formation"}}`:
    '...acts within the Common Pathway of Fibrin Clot Formation...' is **not** correct capitalization
    '...acts within the common pathway of fibrin clot formation...' is correct capitalization

You may slightly rephrase the names of pathways to ensure they are grammatically correct in your generated statement.
"""

entity_extractor_fewshot_examples = """Example input: "The compound kizuta saponin k12 targets prostaglandin G/H synthase 2. The prostaglandin G/H synthase 2 is a gene product of PTGS2, which acts within the pathway of the synthesis of 15-eicosatetraenoic acid derivatives."

Expected response: {{
    "entities": [
        "kizuta saponin k12",
        "prostaglandin G/H synthase 2",
        "PTGS2",
        "synthesis of 15-eicosatetraenoic acid derivatives",
    ]
}}
"""

kg_extractor_fewshot_example = """
1. Example entities: ["kizuta saponin k12", "prostaglandin G/H synthase 2", "PTGS2", "synthesis of 15-eicosatetraenoic acid derivatives"]
1. Example text: "The compound kizuta saponin k12 targets prostaglandin G/H synthase 2. The prostaglandin G/H synthase 2 is a gene product of PTGS2, which acts within the pathway of the synthesis of 15-eicosatetraenoic acid derivatives."

1. Expected output: {{
    "triples": [
        {{"source_node": {{"name": "kizuta saponin k12"}}, "relation": {{"name": "has_target"}}, "target_node": {{"name": "prostaglandin G/H synthase 2"}}}},
        {{"source_node": {{"name": "prostaglandin G/H synthase 2"}}, "relation": {{"name": "gene_product_of"}}, "target_node": {{"name": "PTGS2"}}}},
        {{"source_node": {{"name": "PTGS2"}}, "relation": {{"name": "acts_within"}}, "target_node": {{"name": "synthesis of 15-eicosatetraenoic acid derivatives"}}}},
    ]
}}


2. Example entities: ["xk469", "aldehyde oxidase 1", "toxic liver disease", "neoplasms"]
2. Example text: "The drug xk469 has an effect on aldehyde oxidase 1. Interestingly, alterations in aldehyde oxidase 1 are known to cause several conditions including toxic liver disease, and neoplasms."

2. Expected output: {{
    "triples": [
        {{'source_node': {{'name': 'xk469'}}, 'relation': {{'name': 'is_affecting'}}, 'target_node': {{'name': 'aldehyde oxidase 1'}}}}]"
        {{'source_node': {{'name': 'aldehyde oxidase 1'}}, 'relation': {{'name': 'causes_condition'}}, 'target_node': {{'name': 'toxic liver disease'}}}},
        {{'source_node': {{'name': 'aldehyde oxidase 1'}}, 'relation': {{'name': 'causes_condition'}}, 'target_node': {{'name': 'neoplasms'}}}},
    ]
}}


3. Example entities: ["Aniracetam", "Dopamine D2 receptor", "DRD2", "Magnesium Sulfate", "Paramethadione", "Dihydrocodeine", "Orvepitant"]
3. Example text: "'The drug known as Aniracetam targets the Dopamine D2 receptor, a gene product of DRD2. Aniracetam is known to enhance the efficacy of Magnesium Sulfate, which in turn boosts the effects of Paramethadione, Dihydrocodeine, and Orvepitant. Therefore, caution should be taken when using Aniracetam due to its wide-reaching effects.

3. Expected output: {{
    "triples": [
        {{'source_node': {{'name': 'Aniracetam'}}, 'relation': {{'name': 'has_target'}}, 'target_node': {{'name': 'Dopamine D2 receptor'}}}},
        {{'source_node': {{'name': 'Dopamine D2 receptor'}}, 'relation': {{'name': 'gene_product_of'}}, 'target_node': {{'name': 'DRD2'}}}},
        {{'source_node': {{'name': 'Aniracetam'}}, 'relation': {{'name': 'increase_efficacy'}}, 'target_node': {{'name': 'Magnesium Sulfate'}}}},
        {{'source_node': {{'name': 'Magnesium Sulfate'}}, 'relation': {{'name': 'increase_efficacy'}}, 'target_node': {{'name': 'Paramethadione'}}}},
        {{'source_node': {{'name': 'Magnesium Sulfate'}}, 'relation': {{'name': 'increase_efficacy'}}, 'target_node': {{'name': 'Dihydrocodeine'}}}},
        {{'source_node': {{'name': 'Magnesium Sulfate'}}, 'relation': {{'name': 'increase_efficacy'}}, 'target_node': {{'name': 'Orvepitant'}}}}]"
    ]
}}
"""
