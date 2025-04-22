# Semantic-Knowledge Graph

This repo contains code for generating a semantic-similarity benchmark dataset using knowledge-graphs.

## Installation

### Prerequisites

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.  

1. **Install uv:** If you don't have uv installed, follow the instructions at the provided link.

2. **Clone the repository:**

```bash
git clone https://github.com/EdMorrell/semantic-kg
cd semantic-kg
```

3. **Initialize the project:**

```bash
make init
```

This command will:

* Download all available datasets (*NOTE*: This can be slow. [Download individual datasets](#downloading-individual-datasets) if only requiring certain ones).
* Create and activate a uv virtual environment.
* Install the required Python packages within the virtual environment.


## Running the Code

This project provides two main scripts located in the `bin/` directory:

**1. Create a Subgraph Dataset:**

```bash
python bin/create_subgraph_dataset.py --dataset_name=<dataset_name>
```

Replace `<dataset_name>` with either `prime_kg` or `oregano`.  This script generates a dataset of subgraphs from the specified knowledge graph.

**2. Create a Semantic Benchmark Dataset:**

```bash
python bin/create_semantic_kg.py --dataset_name=<dataset_name>
```

Replace `<dataset_name>` with the name used in the previous step (e.g., `prime_kg`). This script converts the subgraph dataset into a natural-language benchmark dataset.

**Note:** You must configure a language model before running this script (see below).


## Language Model Configuration

This project currently supports OpenAI language models through the Azure OpenAI API.  To configure this:

1. **Copy the example environment file:**

```bash
cp .env.example .env
```

2. **Edit the `.env` file:**  Open the newly created `.env` file and add your Azure OpenAI API key and other required credentials.  Refer to the Azure OpenAI documentation for the specific environment variables needed.



## Configuring Pipelines
All default pipeline configuration files are located under [config/](config/)

[datasets/](config/datasets/) contains dataset configuration files for generating a subgraph dataset from a knowledge-graph

[generation/](config/generation/) contains configuration files for generating natural-language benchmark statements from subgraph datasets.  You can customize these files to modify the data generation process.

## Datasets

### Downloading Individual Datasets

To download the data required to load a single dataset, run:

```bash
make download_dataset DATASET_NAME=<name-of-dataset>
```

### Available Datasets

#### Codex
[Codex](https://github.com/tsafavi/codex) is a general-knowledge, knowledge-graph based on [Wikidata](https://www.wikidata.org/wiki/Wikidata:Main_Page)

#### Oregano
[Oregano](https://gitub.u-bordeaux.fr/erias/oregano) is a drug repositioning knowledge-graph dataset containing information on entities such as drugs, drug-targets, indications and side-effects.

#### PrimeKG
[PrimeKG](https://zitniklab.hms.harvard.edu/projects/PrimeKG/) is a precision medicine knowledge-graph dataset containing relationships between entities such as drugs, diseases and molecular and genetic factors.

#### Globi
[Global Biotics Interactions](https://www.globalbioticinteractions.org/data) is a knowledge-graph describing species interactions (e.g. predator-prey interactions, pollinator-plant interactions)

#### FinDKG
[FinDKG](https://xiaohui-victor-li.github.io/FinDKG/) is a global financial knowledge-graph describing global economic and market trends.


## Tests
All unit-tests are located in the [tests/](tests/) directory. To run the full unit-testing suite run:

```bash
pytest tests/
```

