NODE_TYPES = "['ORG/GOV', 'ORG', 'PERSON', 'SECTOR', 'ORG/REG', 'EVENT', 'ECON_INDICATOR', 'FIN_INSTRUMENT', 'COMP', 'GPE', 'CONCEPT', 'PRODUCT']"
EDGE_TYPES = """['Control', 'Impact', 'Participates_In', 'Relate_To', 'Operate_In', 'Positive_Impact_On', 'Raise', 'Announce', 'Introduce', 'Negative_Impact_On', 'Is_Member_Of', 'Decrease', 'Has', 'Produce', 'Invests_In']"

Relation Definitions:
- Has: Indicates ownership or possession, often of assets or subsidiaries in a financial context.
- Announce: Refers to the formal public declaration of a financial event, product launch, or strategic move.
- Operate_In: Describes the geographical market in which a business entity conducts its operations.
- Introduce: Denotes the first-time introduction of a financial instrument, product, or policy to the market.
- Produce: Specifies the entity responsible for creating a particular product, often in a manufacturing or financial product context.
- Control: Implies authority or regulatory power over monetary policy, financial instruments, or market conditions.
- Participates_In: Indicates active involvement in an event that has financial or economic implications.
- Impact: Signifies a notable effect, either positive or negative, on market trends, financial conditions, or economic indicators.
- Positive_Impact_On: Highlights a beneficial effect on financial markets, economic indicators, or business performance.
- Negative_Impact_On: Underlines a detrimental effect on financial markets, economic indicators, or business performance.
- Relate_To: Points out a connection or correlation with a financial concept, sector, or market trend.
- Is_Member_Of: Denotes membership in a trade group, economic union, or financial consortium.
- Invests_In: Specifies an allocation of capital into a financial instrument, sector, or business entity.
- Raise: Indicates an increase, often referring to capital, interest rates, or production levels in a financial context.
- Decrease: Indicates a reduction, often referring to capital, interest rates, or production levels in a financial context.
"""

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

Relationship Definitions and Examples:
- Has: Indicates ownership or possession, often of assets or subsidiaries in a financial context. Example: Google Has Android.
- Announce: Refers to the formal public declaration of a financial event, product launch, or strategic move. Example: Apple Announces iPhone 13.
- Operate_In: Describes the geographical market in which a business entity conducts its operations. Example: Tesla Operates In China.
- Introduce: Denotes the first-time introduction of a financial instrument, product, or policy to the market. Example: Samsung Introduces Foldable Screen.
- Produce: Specifies the entity responsible for creating a particular product, often in a manufacturing or financial product context. Example: Pfizer Produces Covid-19 Vaccine.
- Control: Implies authority or regulatory power over monetary policy, financial instruments, or market conditions. Example: Federal Reserve Controls Interest Rates.
- Participates_In: Indicates active involvement in an event that has financial or economic implications. Example: United States Participates In G20 Summit.
- Impact: Signifies a notable effect, either positive or negative, on market trends, financial conditions, or economic indicators. Example: Brexit Impacts European Union.
- Positive_Impact_On: Highlights a beneficial effect on financial markets, economic indicators, or business performance. Example: Solar Energy Positive Impact On ESG Ratings.
- Negative_Impact_On: Underlines a detrimental effect on financial markets, economic indicators, or business performance. Example: Covid-19 Negative Impact On Tourism Sector.
- Relate_To: Points out a connection or correlation with a financial concept, sector, or market trend. Example: AI Relates To FinTech Sector.
- Is_Member_Of: Denotes membership in a trade group, economic union, or financial consortium. Example: Germany Is Member Of EU.
- Invests_In: Specifies an allocation of capital into a financial instrument, sector, or business entity. Example: Warren Buffett Invests In Apple.
- Raise: Indicates an increase, often referring to capital, interest rates, or production levels in a financial context. Example: OPEC Raises Oil Production.
- Decrease: Indicates a reduction, often referring to capital, interest rates, or production levels in a financial context. Example: Federal Reserve Decreases Interest Rates.
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
