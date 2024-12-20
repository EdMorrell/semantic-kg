# Semantic-Knowledge Graph

## Installation

1. This package manages dependencies through [uv](https://docs.astral.sh/uv/). Follow the guide to install uv.

To install the package and create a virtual env, run:

```bash
uv sync
```

then to use your virtual env, run:

```bash
source .venv/bin/activate  
```

2. This package also has a pre-commit to ensure consistent formatting. To run on every commit, run:

```bash
pre-commit install
```

## Pre-requisites

### Prime-KG

To generate graphs using PrimeKG you will need to download the relevant data files from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM).

Download `edges.csv` and `nodes.tab` and save them somewhere. The recommended location is in the repo at: `datasets/prime_kg/`


### OpenAI

You will also need to provide API keys to work with OpenAI. Make a copy of [.env.example](.env.example):

```bash
cp .env.example .env
```

And then add the relevant API keys to the `.env` file.


## Running Code

### Prime-KG
The 2 main entry-points are located under `/bin`:

1. [create_primekg_subgraph_dataset.py](bin/create_primekg_subgraph_dataset.py): Will generate a dataset of knowledge-graphs and perturbed knowledge graphs
1. [create_primekg.py](bin/create_primekg.py): Will take the dataset generated in `1.` and use it to generate responses with a language-model
