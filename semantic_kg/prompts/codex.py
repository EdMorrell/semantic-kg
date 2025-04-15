from pathlib import Path
from typing import Optional

from semantic_kg.prompts import PromptConfig, utils
from semantic_kg.datasets.codex import CodexLoader

# flake8: noqa: E501


EDGE_TYPES = "['country of citizenship', 'country', 'occupation', 'place of birth', 'member of political party', 'educated at', 'genre', 'member of', 'located in the administrative terroritorial entity', 'languages spoken, written, or signed', 'religion', 'instrument', 'sibling', 'place of death', 'shares border with', 'spouse', 'place of burial', 'cast member', 'record label', 'field of work', 'employer', 'influenced by', 'location of formation', 'diplomatic relation', 'cause of death', 'country of origin', 'residence', 'airline hub', 'official language', 'narrative location', 'capital', 'ethnic group', 'member of sports team', 'language of work or name', 'time period', 'headquarters location', 'child', 'sport', 'medical condition', 'movement', 'director', 'uses', 'founded by', 'parent organization', 'continent', 'occupant', 'mountain range', 'symptoms', 'part of', 'publisher', 'drug used for treatment', 'industry', 'named after', 'unmarried partner', 'airline alliance', 'creator', 'legal form', 'author', 'chairperson', 'health specialty', 'architect', 'chief executive officer', 'product or material produced', 'architectural style', 'legislative body', 'practiced by', 'foundational text', 'studies', 'use']"


codex_fewshot_examples = """
triple = [
    {{'source_node': {{'name': 'Norway'}}, 'relation': {{'name': 'diplomatic relation'}}, 'target_node': {{'name': 'Colombia'}}}},
    {{'source_node': {{'name': 'Norway'}}, 'relation': {{'name': 'shares border with'}}, 'target_node': {{'name': 'Russia'}}}},
    {{'source_node': {{'name': 'Norway'}}, 'relation': {{'name': 'member of'}}, 'target_node': {{'name': 'Organisation for the Prohibition of Chemical Weapons'}}}},
    {{'source_node': {{'name': 'Russia'}}, 'relation': {{'name': 'diplomatic relation'}}, 'target_node': {{'name': 'Ethiopia'}}}},
    {{'source_node': {{'name': 'Colombia'}}, 'relation': {{'name': 'member of'}}, 'target_node': {{'name': 'Universal Postal Union'}}}},
    {{'source_node': {{'name': 'Colombia'}}, 'relation': {{'name': 'member of'}}, 'target_node': {{'name': 'Andean Community'}}}},
    {{'source_node': {{'name': 'Colombia'}}, 'relation': {{'name': 'member of'}}, 'target_node': {{'name': 'Organisation for the Prohibition of Chemical Weapons'}}}},
]

You could say:
"Norway has diplomatic relations with Colombia and Russia. Russia is also known to have diplomatic relations with Ethiopia. Both Norway and Colombia are members of the Organisation for the Prohibition of Chemical weapons, while Colombia is also a member of the Universal Postal Union and the Andean Community."

or given the triple:

triple = [
    {{'source_node': {{'name': 'Renate Axt'}}, 'relation': {{'name': 'occupation'}}, 'target_node': {{'name': 'writer'}}}},
    {{'source_node': {{'name': 'Renate Axt'}}, 'relation': {{'name': 'place of birth'}}, 'target_node': {{'name': 'Darmstadt'}}}},
    {{'source_node': {{'name': 'Darmstadt'}}, 'relation': {{'name': 'country'}}, 'target_node': {{'name': 'Germany'}}}}
    {{'source_node': {{'name': 'Germany'}}, 'relation': {{'name': 'shares border with'}}, 'target_node': {{'name': 'Austria'}}}},
    {{'source_node': {{'name': 'Germany'}}, 'relation': {{'name': 'shares border with'}}, 'target_node': {{'name': 'Czechoslovakia'}}}},
]

You could say:
"Renate Axt was a writer, born in Darmstadt in Germany. Germany shares a border with Austria and Czechoslovakia."

or given the triple:

triple = [
    {{'source_node': {{'name': 'Frederick William II of Prussia'}}, 'relation': {{'name': 'child'}}, 'target_node': {{'name': 'Friedrich Wilhelm, Count Brandenburg'}}}},
    {{'source_node': {{'name': 'Frederick William II of Prussia'}}, 'relation': {{'name': 'member of'}}, 'target_node': {{'name': 'Saint Petersburg Academy of Sciences'}}}},
    {{'source_node': {{'name': 'Friedrich Wilhelm, Count Brandenburg'}}, 'relation': {{'name': 'place of birth'}}, 'target_node': {{'name': 'Berlin'}}}},
    {{'source_node': {{'name': 'Friedrich Wilhelm, Count Brandenburg'}}, 'relation': {{'name': 'occupation'}}, 'target_node': {{'name': 'politician'}}}},
    {{'source_node': {{'name': 'Saint Petersburg Academy of Sciences'}}, 'relation': {{'name': 'country'}}, 'target_node': {{'name': 'Russia'}}}},
    {{'source_node': {{'name': 'Saint Petersburg Academy of Sciences'}}, 'relation': {{'name': 'located in the administrative terroritorial entity'}}, 'target_node': {{'name': 'Saint Petersburg'}}}},
]

You could say:
"Frederick William II of Prussia was the child of Friedrich Wilhelm, Count Brandenburg, a politician who was born in Berlin. Frederick William II was a member of the Saint Petersburg Academy of Sciences, a Russian institute, located in Saint Petersburg"
"""


codex_prompt_rules = """In your final response, do NOT put name of any node or relation in quotes.
For example for the node `{{"name": "The Godfather"}}`:
    'The movie  "The Godfather"...' is **not** allowed
    '... starred in the movie "The Godfather"' is **not** allowed
    '... alongside Marlon Brando in 'The Godfather' is allowed

You may slightly rephrase the names of edges to ensure they are grammatically correct and produce a fluent, coherent sounding statement
"""

entity_extractor_fewshot_examples = """Example input: "Renate Axt was a writer, born in Darmstadt in Germany. Germany shares a border with Austria and Czechoslovakia."

Expected response: {{
    entities: [
        "Renate Axt",
        "Darmstadt",
        "Germany",
        "Austria",
        "Czechoslovakia",
    ]
}}
"""

kg_extractor_fewshot_examples = """
1. Example entities: ["Renate Axt", "Darmstadt", "Germany", "Austria", "Czechoslovakia"]
1. Example text: "Renate Axt was a writer, born in Darmstadt in Germany. Germany shares a border with Austria and Czechoslovakia."

1. Expected output: {{
    "triples": [
        {{'source_node': {{'name': 'Renate Axt'}}, 'relation': {{'name': 'occupation'}}, 'target_node': {{'name': 'writer'}}}},
        {{'source_node': {{'name': 'Renate Axt'}}, 'relation': {{'name': 'place of birth'}}, 'target_node': {{'name': 'Darmstadt'}}}},
        {{'source_node': {{'name': 'Darmstadt'}}, 'relation': {{'name': 'country'}}, 'target_node': {{'name': 'Germany'}}}}
        {{'source_node': {{'name': 'Germany'}}, 'relation': {{'name': 'shares border with'}}, 'target_node': {{'name': 'Austria'}}}},
        {{'source_node': {{'name': 'Germany'}}, 'relation': {{'name': 'shares border with'}}, 'target_node': {{'name': 'Czechoslovakia'}}}},
    ]
}}


2. Example entities: ["Rome", "Roman Republic", "ancient Rome", "Western Roman Empire", "Persian Empire", "classical antiquity", "Latin"]
2. Example text: "Rome has been located in a range of administrative terroritorial entities including the Roman Republic, ancient Rome and the Western Roman Empire. Ancient Rome, which shared a border with the Perian Empire, existed in the time period of classical antiquity. Within the Western Roman Empire, the official language was Latin."

2. Expected output: {{
    "triples": [
        {{'source_node': {{'name': 'Rome'}}, 'relation': {{'name': 'located in the administrative terroritorial entity'}}, 'target_node': {{'name': 'Roman Republic'}}}},
        {{'source_node': {{'name': 'Rome'}}, 'relation': {{'name': 'located in the administrative terroritorial entity'}}, 'target_node': {{'name': 'ancient Rome'}}}},
        {{'source_node': {{'name': 'Rome'}}, 'relation': {{'name': 'located in the administrative terroritorial entity'}}, 'target_node': {{'name': 'Western Roman Empire'}}}},
        {{'source_node': {{'name': 'ancient Rome'}}, 'relation': {{'name': 'shares border with'}}, 'target_node': {{'name': 'Persian Empire'}}}},
        {{'source_node': {{'name': 'ancient Rome'}}, 'relation': {{'name': 'time period'}}, 'target_node': {{'name': 'classical antiquity'}}}},
        {{'source_node': {{'name': 'Western Roman Empire'}}, 'relation': {{'name': 'official language'}}, 'target_node': {{'name': 'Latin'}}}},
    ]
}}



3. Example entities: ["G.I. Joe: The Rise of Cobra", "Lee Byung-hun", "Marlon Wayans", "film director", "New York City", "Howard University", "film actor", "Seoul", "Korean"]
3. Example text: "G.I. Joe: The Rise of Cobra, a film that takes place in Tokyo, starred Lee Byung-hun and Marlon Wayans. Wayans, also a film-director was born in New York City and attended Howard University. Meanwhile Lee Byung-hun is an actor born in Seoul who speaks Korean."
[
    {'source_node': {'name': 'G.I. Joe: The Rise of Cobra'}, 'relation': {'name': 'cast member'}, 'target_node': {'name': 'Lee Byung-hun'}},
    {'source_node': {'name': 'G.I. Joe: The Rise of Cobra'}, 'relation': {'name': 'cast member'}, 'target_node': {'name': 'Marlon Wayans'}},
    {'source_node': {'name': 'G.I. Joe: The Rise of Cobra'}, 'relation': {'name': 'narrative location'}, 'target_node': {'name': 'Tokyo'}},
    {'source_node': {'name': 'Marlon Wayans'}, 'relation': {'name': 'occupation'}, 'target_node': {'name': 'film director'}},
    {'source_node': {'name': 'Marlon Wayans'}, 'relation': {'name': 'place of birth'}, 'target_node': {'name': 'New York City'}},
    {'source_node': {'name': 'Marlon Wayans'}, 'relation': {'name': 'educated at'}, 'target_node': {'name': 'Howard University'}},
    {'source_node': {'name': 'Lee Byung-hun'}, 'relation': {'name': 'occupation'}, 'target_node': {'name': 'film actor'}},
    {'source_node': {'name': 'Lee Byung-hun'}, 'relation': {'name': 'place of birth'}, 'target_node': {'name': 'Seoul'}},
    {'source_node': {'name': 'Lee Byung-hun'}, 'relation': {'name': 'languages spoken, written, or signed'}, 'target_node': {'name': 'Korean'}},
]
"""


def get_all_node_types(data_dir: Optional[str | Path] = None) -> str:
    """Computes all node-types directly from the dataset

    Parameters
    ----------
    data_dir : Optional[str  |  Path], optional
        Directory where codex data is located, by default None

    Returns
    -------
    str
        String representing list of nodes
    """
    if not data_dir:
        root_dir = Path(__file__).parent.parent.parent
        data_dir = root_dir / "datasets" / "codex"
    loader = CodexLoader(data_dir=data_dir)
    triple_df = loader.load()
    head_types = triple_df["head_type"].unique().tolist()
    tail_types = triple_df["tail_type"].unique().tolist()
    node_types = list(set(head_types + tail_types))

    return str(node_types)


CODEX_PROMPT_CONFIG = PromptConfig(
    response_model_prompt=utils.get_default_prompt_template(
        fewshot_examples=codex_fewshot_examples,
        prompt_rules=codex_prompt_rules,
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
