# Revosax Embeddings

This project builds a legal question-answering system based on German law texts. It processes legislative documents, generates natural language questions, trains semantic embeddings, and evaluates retrieval-based QA pipelines.

## Features

* Preprocessing of German legal documents from Markdown files
* Automatic generation of citizen-style legal questions using GPT-4
* Training-ready dataset creation for embedding models
* Evaluation of multiple multilingual embedding models (e.g., IBM Granite)
* Integration with ChromaDB for vector search
* RAG-style (retrieval-augmented generation) answering pipeline

## Installation

1. **Create a virtual environment:**

```bash
python -m venv .venv
```

2. **Activate the virtual environment:**

* On Windows:

```bash
.venv\Scripts\activate
```

* On macOS and Linux:

```bash
source .venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

## Structure

* `data_import/`

  * Scripts and prompts for generating training data (questions from law texts)
* `examples/rag/`

  * Notebooks for running retrieval-augmented generation pipelines
* `evaluation/`

  * Evaluation scripts using SentenceTransformer and IR metrics
* `data/`

  * Contains raw and processed legal documents and training data

## Usage

### Generate Training Data

Use `build-training-data.py` to parse Markdown law texts and generate questions via GPT API.

```bash
python data_import/build-training-data.py
```

This script splits legal content into chunks, applies a prompt, and saves the results in a CSV.

### Evaluate Embeddings

Run `eval.ipynb` to test different embedding models (e.g., IBM Granite, MPNet) and compute metrics like accuracy\@k, NDCG, MRR, MAP.

### RAG-based Answering

Use `retrival_chroma.ipynb` to embed chunks, store them in ChromaDB, and run a QA pipeline using a system prompt and GPT.

## License

This project is licensed under the MIT License. See `LICENSE` file for details.


## Acknowledgements

* OpenAI GPT API
* HuggingFace Transformers & Datasets
* ChromaDB
* IBM Granite Embeddings
