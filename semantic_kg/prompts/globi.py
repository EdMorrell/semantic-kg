from pathlib import Path
from typing import Optional

from semantic_kg.prompts import PromptConfig, utils
from semantic_kg.datasets import GlobiLoader

# flake8: noqa: E501

EDGE_TYPES = "['parasiteOf', 'hasHost', 'eats', 'preysOn', 'pollinates', 'pathogenOf', 'visitsFlowersOf', 'hasVector', 'rootparasiteOf', 'endoparasiteOf', 'interactsWith', 'kills', 'createsHabitatFor', 'parasitoidOf', 'hasRoost', 'coRoostsWith', 'ecologicallyRelatedTo', 'epiphyteOf', 'commensalistOf', 'mutualistOf', 'providesNutrientsFor', 'ectoparasiteOf', 'coOccursWith', 'hasHabitat', 'symbiontOf', 'kleptoparasiteOf', 'adjacentTo', 'allelopathOf', 'laysEggsIn', 'visits', 'hyperparasiteOf', 'laysEggsOn', 'endoparasitoidOf', 'livesOn', 'guestOf', 'livesInsideOf', 'ectoParasitoid', 'livesNear', 'livesUnder', 'inhabits', 'hasDispersalVector']"


globi_fewshot_examples = """
triple = [
    {{'source_node': {{'name': 'Pinus jeffreyi'}}, 'relation': {{'name': 'interactsWith'}}, 'target_node': {{'name': 'Betula occidentalis'}}}},
    {{'source_node': {{'name': 'Pinus jeffreyi'}}, 'relation': {{'name': 'interactsWith'}}, 'target_node': {{'name': 'Wyethia mollis'}}}},
    {{'source_node': {{'name': 'Wyethia mollis'}}, 'relation': {{'name': 'interactsWith'}}, 'target_node': {{'name': 'Collomia heterophylla'}}}}
]

You could say:
"Pinus jeffreyi interacts with several species including Betula occidentalis and Wyethia mollis. Wyethia mollis in turn interacts with Collomia heterophylla"


triple = [
    {{'source_node': {{'name': 'Neotoma mexicana'}}, 'relation': {{'name': 'interactsWith'}}, 'target_node': {{'name': 'Lynx rufus'}}}},
    {{'source_node': {{'name': 'Neotoma mexicana'}}, 'relation': {{'name': 'interactsWith'}}, 'target_node': {{'name': 'Canis latrans'}}}}
    {{'source_node': {{'name': 'Canis latrans'}}, 'relation': {{'name': 'eats'}}, 'target_node': {{'name': 'Prunus serotina'}}}},
    {{'source_node': {{'name': 'Canis latrans'}}, 'relation': {{'name': 'eats'}}, 'target_node': {{'name': 'Sylvilagus cunicularius'}}}},
    {{'source_node': {{'name': 'Canis latrans'}}, 'relation': {{'name': 'coOccursWith'}}, 'target_node': {{'name': 'Panthera leo'}}}},
    {{'source_node': {{'name': 'Crotalus pricei'}}, 'relation': {{'name': 'preysOn'}}, 'target_node': {{'name': 'Sceloporus jarrovii'}}}},
    {{'source_node': {{'name': 'Crotalus pricei'}}, 'relation': {{'name': 'preysOn'}}, 'target_node': {{'name': 'Neotoma mexicana'}}}},
    {{'source_node': {{'name': 'Crotalus pricei'}}, 'relation': {{'name': 'preysOn'}}, 'target_node': {{'name': 'Junco phaeonotus'}}}},
]

You could say:
"Crotalus pricei is a species known to prey on several species including, Sceloporus jarrovii, Junco phaeonotus, and Neotoma mexicana. Neotoma mexicana have interactions with several species including Lynx rufus and Canis latrans. Canis latrans are known to co-occur with Panthera leo and eat Prunus serotina and Sylvilagus cuncicularius."

triple = [
    {{'source_node': {{'name': 'Chromatomyia erigerontophaga'}}, 'relation': {{'name': 'visitsFlowersOf'}}, 'target_node': {{'name': 'Potentilla nivea'}}}},
    {{'source_node': {{'name': 'Chromatomyia erigerontophaga'}}, 'relation': {{'name': 'pollinates'}}, 'target_node': {{'name': 'Erigeron compositus'}}}},
    {{'source_node': {{'name': 'Potentilla nivea'}}, 'relation': {{'name': 'interactsWith'}}, 'target_node': {{'name': 'Salix arctica'}}}},
    {{'source_node': {{'name': 'Potentilla nivea'}}, 'relation': {{'name': 'interactsWith'}}, 'target_node': {{'name': 'Draba nivalis'}}}},
    {{'source_node': {{'name': 'Erigeron compositus'}}, 'relation': {{'name': 'interactsWith'}}, 'target_node': {{'name': 'Populus tremuloides'}}}},
    {{'source_node': {{'name': 'Erigeron compositus'}}, 'relation': {{'name': 'interactsWith'}}, 'target_node': {{'name': 'Luetkea pectinata'}}}}
]

You could say:
"Chromatomyia erigerontophaga is a species known to visit the flowers of Potentilla nivea which in turn have interactions with Salix arctica and Draba nivalis. C. erigerontophaga also pollinate Erigeron compositus. This flower is known to interact with Populus tremuloides and Luetkea pectinata."
"""

globi_prompt_rules = """In your final response, do NOT put name of any node or relation in quotes.
For example for the node `{{"name": "Baccharis sarothroides"}}`:
    'The "Baccharis sarothroides"...' is **not** allowed
    '... interacts with the plant "Baccharis sarothroides"' is **not** allowed
    "The plant 'Baccharis sarothroides'..." is **not** allowed
"""

entity_extractor_fewshot_examples = """Example input: "Chromatomyia erigerontophaga is a species known to visit the flowers of Potentilla nivea which in turn have interactions with Salix arctica and Draba nivalis. C. erigerontophaga also pollinate Erigeron compositus. This flower is known to interact with Populus tremuloides and Luetkea pectinata."

Expected response: {{
    "entities": [
        "Chromatomyia erigerontophaga",
        "Potentilla nivea",
        "Salix arctica",
        "Draba nivalis",
        "Erigeron compositus",
        "Populus tremuloides",
        "Luetkea pectinata",
    ]
}}
"""

kg_extractor_fewshot_examples = """
1. Example entities: ["Chromatomyia erigerontophaga", "Potentilla nivea", "Salix arctica", "Draba nivalis", "Erigeron compositus", "Populus tremuloides", "Luetkea pectinata"]
1. Example text: "Chromatomyia erigerontophaga is a species known to visit the flowers of Potentilla nivea which in turn have interactions with Salix arctica and Draba nivalis. C. erigerontophaga also pollinate Erigeron compositus. This flower is known to interact with Populus tremuloides and Luetkea pectinata."

1. Expected output: {{
    "triples": [
        {{'source_node': {{'name': 'Chromatomyia erigerontophaga'}}, 'relation': {{'name': 'visitsFlowersOf'}}, 'target_node': {{'name': 'Potentilla nivea'}}}},
        {{'source_node': {{'name': 'Chromatomyia erigerontophaga'}}, 'relation': {{'name': 'pollinates'}}, 'target_node': {{'name': 'Erigeron compositus'}}}},
        {{'source_node': {{'name': 'Potentilla nivea'}}, 'relation': {{'name': 'interactsWith'}}, 'target_node': {{'name': 'Salix arctica'}}}},
        {{'source_node': {{'name': 'Potentilla nivea'}}, 'relation': {{'name': 'interactsWith'}}, 'target_node': {{'name': 'Draba nivalis'}}}},
        {{'source_node': {{'name': 'Erigeron compositus'}}, 'relation': {{'name': 'interactsWith'}}, 'target_node': {{'name': 'Populus tremuloides'}}}},
        {{'source_node': {{'name': 'Erigeron compositus'}}, 'relation': {{'name': 'interactsWith'}}, 'target_node': {{'name': 'Luetkea pectinata'}}}}
    ]
}}

2. Example entities: ["Pinus jeffreyi", "Betula occidentalis", "Wyethia mollis", "Collomia heterophylla"]
2. Example text: "Pinus jeffreyi interacts with several species including Betula occidentalis and Wyethia mollis. Wyethia mollis in turn interacts with Collomia heterophylla."

2. Expected output: {{
    "triples": [
        {{'source_node': {{'name': 'Pinus jeffreyi'}}, 'relation': {{'name': 'interactsWith'}}, 'target_node': {{'name': 'Betula occidentalis'}}}},
        {{'source_node': {{'name': 'Pinus jeffreyi'}}, 'relation': {{'name': 'interactsWith'}}, 'target_node': {{'name': 'Wyethia mollis'}}}},
        {{'source_node': {{'name': 'Wyethia mollis'}}, 'relation': {{'name': 'interactsWith'}}, 'target_node': {{'name': 'Collomia heterophylla'}}}}
    ]
}}

3. Example entities: ["Neotoma mexicana", "Lynx rufus", "Canis latrans", "Prunus serotina", "Sylvilagus cunicularius", "Panthera leo", "Crotalus pricei", "Sceloporus jarrovii", "Junco phaeonotus"]
3. Example text: "Crotalus pricei is a species known to prey on several species including, Sceloporus jarrovii, Junco phaeonotus, and Neotoma mexicana. Neotoma mexicana have interactions with several species including Lynx rufus and Canis latrans. Canis latrans are known to co-occur with Panthera leo and eat Prunus serotina and Sylvilagus cuncicularius."

3. Expected output: {{
    "triples": [
        {{'source_node': {{'name': 'Neotoma mexicana'}}, 'relation': {{'name': 'interactsWith'}}, 'target_node': {{'name': 'Lynx rufus'}}}},
        {{'source_node': {{'name': 'Neotoma mexicana'}}, 'relation': {{'name': 'interactsWith'}}, 'target_node': {{'name': 'Canis latrans'}}}}
        {{'source_node': {{'name': 'Canis latrans'}}, 'relation': {{'name': 'eats'}}, 'target_node': {{'name': 'Prunus serotina'}}}},
        {{'source_node': {{'name': 'Canis latrans'}}, 'relation': {{'name': 'eats'}}, 'target_node': {{'name': 'Sylvilagus cunicularius'}}}},
        {{'source_node': {{'name': 'Canis latrans'}}, 'relation': {{'name': 'coOccursWith'}}, 'target_node': {{'name': 'Panthera leo'}}}},
        {{'source_node': {{'name': 'Crotalus pricei'}}, 'relation': {{'name': 'preysOn'}}, 'target_node': {{'name': 'Sceloporus jarrovii'}}}},
        {{'source_node': {{'name': 'Crotalus pricei'}}, 'relation': {{'name': 'preysOn'}}, 'target_node': {{'name': 'Neotoma mexicana'}}}},
        {{'source_node': {{'name': 'Crotalus pricei'}}, 'relation': {{'name': 'preysOn'}}, 'target_node': {{'name': 'Junco phaeonotus'}}}},
    ]
}}
"""


def get_all_node_types(data_dir: Optional[str | Path] = None) -> str:
    """Computes all node-types directly from the dataset

    Parameters
    ----------
    data_dir : Optional[str  |  Path], optional
        Directory where globi data is located, by default None

    Returns
    -------
    str
        String representing list of nodes
    """
    if not data_dir:
        root_dir = Path(__file__).parent.parent.parent
        data_dir = root_dir / "datasets" / "globi"
    loader = GlobiLoader(data_dir=data_dir)
    triple_df = loader.load()
    src_types = triple_df["sourceTaxonFamilyName"].unique().tolist()
    target_types = triple_df["targetTaxonFamilyName"].unique().tolist()
    node_types = list(set(src_types + target_types))

    return str(node_types)


GLOBI_PROMPT_CONFIG = PromptConfig(
    response_model_prompt=utils.get_default_prompt_template(
        fewshot_examples=globi_fewshot_examples,
        prompt_rules=globi_prompt_rules,
        directed=True,
    ),
    entity_extractor_scorer_prompt=utils.get_entity_extractor_system_prompt(
        node_types=get_all_node_types(),
        entity_extractor_fewshot_examples=entity_extractor_fewshot_examples,
    ),
    kg_extractor_scorer_prompt=utils.get_kg_extractor_system_prompt(
        edge_types=EDGE_TYPES,
        fewshot_examples=kg_extractor_fewshot_examples,
        incl_valid_directions=False,
    ),
)
