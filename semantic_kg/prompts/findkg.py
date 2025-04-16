NODE_TYPES = "['ORG/GOV', 'ORG', 'PERSON', 'SECTOR', 'ORG/REG', 'EVENT', 'ECON_INDICATOR', 'FIN_INSTRUMENT', 'COMP', 'GPE', 'CONCEPT', 'PRODUCT']"
EDGE_TYPES = "['Control', 'Impact', 'Participates_In', 'Relate_To', 'Operate_In', 'Positive_Impact_On', 'Raise', 'Announce', 'Introduce', 'Negative_Impact_On', 'Is_Member_Of', 'Decrease', 'Has', 'Produce', 'Invests_In']"

# flake8: noqa: E501


findkg_fewshot_examples = """
triple = [
    {{'source_node': {{'name': 'U.S. Air Force'}}, 'relation': {{'name': 'Control'}}, 'target_node': {{'name': 'Asia and Europe'}}}},
    {{'source_node': {{'name': 'U.S. Air Force'}}, 'relation': {{'name': 'Operate_In'}}, 'target_node': {{'name': 'Afghanistan'}}}},
    {{'source_node': {{'name': 'Afghanistan'}}, 'relation': {{'name': 'Has'}}, 'target_node': {{'name': 'government'}}}},
]

You could say:
"The U.S. Air Force controls Asia and Europe, and operates in Afghanistan. Afghanistan has a government."

or given the triple:

triple = [
    {{'source_node': {{'name': 'Italian Debt'}}, 'relation': {{'name': 'Impact'}}, 'target_node': {{'name': 'Investors'}}}},
    {{'source_node': {{'name': 'Italian Debt'}}, 'relation': {{'name': 'Impact'}}, 'target_node': {{'name': 'Italian Government'}}}},
    {{'source_node': {{'name': 'Italian Debt'}}, 'relation': {{'name': 'Relate_To'}}, 'target_node': {{'name': 'European Central Bank'}}}},
    {{'source_node': {{'name': 'Investors'}}, 'relation': {{'name': 'Impact'}}, 'target_node': {{'name': 'Yuan'}}}},
]

You could say:
"Italian debt is related to policies of the European Central Bank. This debt has an impact on the Italian Government in addition to investors. Investors may also subsequently impact the value of Yuan."

or given the triple:

triple = [
    {{'source_node': {{'name': 'Federal Reserve System'}}, 'relation': {{'name': 'Impact'}}, 'target_node': {{'name': 'Gold'}}}},
    {{'source_node': {{'name': 'Federal Reserve System'}}, 'relation': {{'name': 'Control'}}, 'target_node': {{'name': 'Expenses'}}}},
    {{'source_node': {{'name': 'Federal Reserve System'}}, 'relation': {{'name': 'Control'}}, 'target_node': {{'name': 'The U.S. Economy'}}}},
    {{'source_node': {{'name': 'The U.S. Economy'}}, 'relation': {{'name': 'Relate_To'}}, 'target_node': {{'name': 'Gross Domestic Product'}}}},
    {{'source_node': {{'name': 'The U.S. Economy'}}, 'relation': {{'name': 'Relate_To'}}, 'target_node': {{'name': 'U.S. stocks'}}}},
    {{'source_node': {{'name': 'Expenses'}}, 'relation': {{'name': 'Positive_Impact_On'}}, 'target_node': {{'name': 'Gold'}}}},
]

You could say:
"The Federal Reserve System controls expenses, which can have a positive impact on Gold, an asset also impacted by the Federal Reserve System. Additionally this system controls the U.S. Economy which has a relationship with Gross Domestic Product and U.S. stocks."
"""


findkg_prompt_rules = """In your final response, do NOT put name of any node or relation in quotes.
For example for the node `{{"name": "Federal Reserve System"}}`:
    'The "Federal Reserve System"...' is **not** allowed
    '... is controlled by the "Federal Reserve System"' is **not** allowed
    '... is controlled by the 'Federal Reserve System' is **not** allowed
"""

entity_extractor_fewshot_examples = """Example input: "Italian debt is related to policies of the European Central Bank. This debt has an impact on the Italian Government in addition to investors. Investors may also subsequently impact the value of Yuan."

Expected response: {{
    "entities": [
        "Italian debt",
        "European Central Bank",
        "Italian Government",
        "investors",
        "Yuan",
    ]
}}
"""

kg_extractor_fewshot_examples = """
1. Example entities = ["U.S. Air Force", "Asia and Europe", "Afghanistan", "government"]
1. Example text = "The U.S. Air Force controls Asia and Europe, and operates in Afghanistan. Afghanistan has a government."

1. Expected output: {{
    "triples": [
        {{'source_node': {{'name': 'U.S. Air Force'}}, 'relation': {{'name': 'Control'}}, 'target_node': {{'name': 'Asia and Europe'}}}},
        {{'source_node': {{'name': 'U.S. Air Force'}}, 'relation': {{'name': 'Operate_In'}}, 'target_node': {{'name': 'Afghanistan'}}}},
        {{'source_node': {{'name': 'Afghanistan'}}, 'relation': {{'name': 'Has'}}, 'target_node': {{'name': 'government'}}}},
    ]
}}

2. Example entities = ["Tax Cut", "consumer spending", "investment", "Economic indicators", "The U.S. Economy"]
2. Example text = "Tax cuts can impact investment but also have a positive impact on consumer spending which relates to important economic indicators and can impact the U.S. economy."

2. Expected output: {{
    "triples": [
        {{'source_node': {{'name': 'Tax Cut'}}, 'relation': {{'name': 'Positive_Impact_On'}}, 'target_node': {{'name': 'Consumer Spending'}}}},
        {{'source_node': {{'name': 'Tax Cut'}}, 'relation': {{'name': 'Impact'}}, 'target_node': {{'name': 'investment'}}}},
        {{'source_node': {{'name': 'Consumer Spending'}}, 'relation': {{'name': 'Relate_To'}}, 'target_node': {{'name': 'Economic indicators'}}}},
        {{'source_node': {{'name': 'Consumer Spending'}}, 'relation': {{'name': 'Impact'}}, 'target_node': {{'name': 'The U.S. Economy'}}}},
    ]
}}

3. Example entities: ["Federal Reserve System", "Gold", "Expenses", "The U.S. Economy", "Gross Domestic Product", "U.S. stocks"]
3. Example text = "The Federal Reserve System controls expenses, which can have a positive impact on Gold, an asset also impacted by the Federal Reserve System. Additionally this system controls the U.S. Economy which has a relationship with Gross Domestic Product and U.S. stocks."

3. Expected output: {{
    "triples": [
        {{'source_node': {{'name': 'Federal Reserve System'}}, 'relation': {{'name': 'Impact'}}, 'target_node': {{'name': 'Gold'}}}},
        {{'source_node': {{'name': 'Federal Reserve System'}}, 'relation': {{'name': 'Control'}}, 'target_node': {{'name': 'Expenses'}}}},
        {{'source_node': {{'name': 'Federal Reserve System'}}, 'relation': {{'name': 'Control'}}, 'target_node': {{'name': 'The U.S. Economy'}}}},
        {{'source_node': {{'name': 'The U.S. Economy'}}, 'relation': {{'name': 'Relate_To'}}, 'target_node': {{'name': 'Gross Domestic Product'}}}},
        {{'source_node': {{'name': 'The U.S. Economy'}}, 'relation': {{'name': 'Relate_To'}}, 'target_node': {{'name': 'U.S. stocks'}}}},
        {{'source_node': {{'name': 'Expenses'}}, 'relation': {{'name': 'Positive_Impact_On'}}, 'target_node': {{'name': 'Gold'}}}},
    ]
]
"""
