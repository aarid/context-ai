# README

## Overview

This application is a Retrieval-Augmented Generation (RAG) model application designed to generate context-aware responses by leveraging the **Hugging Face** inference API for embeddings and **LangChain** Community's ChromaDB for data storage. The application retrieves context from a local database and uses a local model to generate responses.

## Project Structure

```
├── README.md
├── src
│   ├── main.py
│   ├── data_loader.py
│   └── embedding_function.py
├── data
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
└── DATA_DB
```

## Data Loading and Storage

The application processes PDF files stored in the `data` directory and saves them in a local SQLite database (`DATA_DB`) using ChromaDB. This process is managed by the `data_loader.py` script. Ensure you create a folder named `data` and place your PDF files in this directory.

## Embedding

For embedding data, the application utilizes the Hugging Face inference API. Ensure the API key for the Hugging Face inference API (`HF_INFERENCE_API_KEY`) is set as an environment variable.

## Model

The application uses a local model (e.g., `phi3`) with the LM Studio API server running on the local machine. The model fetches context from the `DATA_DB` and generates responses based on this context.

## Setup

This project uses a Conda environment for managing dependencies. You can create and activate the Conda environment using the provided `env.yml` file with the following commands:

```bash
conda env create -f env.yml
conda activate rag-bot
```

### Hugging Face Inference API Key
 #### Get your API Token
   - Go to the Hugging Face website: https://huggingface.co/
   -  Register or Login.
   - Get a User Access or API token in your Hugging Face profile settings.

You should see a token hf_xxxxx (old tokens are api_XXXXXXXX or api_org_XXXXXXX).

After creating an account and acquiring your key, set it as an environment variable on Windows:

1. Open the Start Menu and search for "Environment Variables".
2. Click "Edit the system environment variables".
3. In the System Properties window, click "Environment Variables".
4. Click "New" under "User variables".
5. Set the variable name to `HF_INFERENCE_API_KEY` and the value to your API key.
6. Click OK to save the changes.

### LM Studio

The application requires the LM Studio API server with a local model. Download and install LM Studio from the [official website](https://lmstudio.ai/). Follow the provided instructions to set up and run LM Studio on your machine.

## Running the Application

To start the application, run the `main.py` script. This script initiates a loop where you can input queries, and the application will generate responses based on the retrieved context from the `DATA_DB`.

```bash
python src/main.py
```
