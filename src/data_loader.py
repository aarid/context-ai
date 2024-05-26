import argparse
import os
import shutil
import time

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from embedding_function import get_embedding_bedrock_function
from langchain_community.vectorstores import Chroma

from embedding_function import *

DB_PATH = "../MATH_DB"
DATA_PATH = "../data"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def load_documents():
    """
    Load the documents from the PDF files in the data directory.
    """
    # Create the directory loader
    loader = PyPDFDirectoryLoader(DATA_PATH)

    return loader.load()


def split_documents(documents):
    """
    Split the documents into characters.
    """
    # Create the splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )

    return splitter.split_documents(documents)


def calculate_chunks_ids(chunks):
    """
    Calculate the chunk IDs.
    # This will create IDs like "data/algebra.pdf:14:2"
    # Page Source : Page Number : Chunk Index
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def update_database(chunks):
    """
    Update the database with the documents.
    """

    model_name = "BAAI/bge-m3"

    # Load the existing database
    db = Chroma(persist_directory=DB_PATH, embedding_function=get_huggingface_embedding_function(model_name))

    # Calculate page IDs
    chunks_with_ids = calculate_chunks_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = []
        total_chunks = len(new_chunks)
        for i, chunk in enumerate(new_chunks):
            new_chunk_ids.append(chunk.metadata["id"])
            # time.sleep(1)  # sleep for 1 second
            progress = (i + 1) / total_chunks * 100
            print(f"Adding document: {chunk.metadata['id']}. Progress: {progress:.2f}%")
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def clear_database():
    """
    Clear the database.
    """
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print("🗑️ Cleared the database")


def main():
    """
    Main function.
    """
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Update the database with the documents in the data directory.")
    parser.add_argument("--clear", action="store_true", help="Clear the database before updating it.")
    args = parser.parse_args()

    # Load the documents
    documents = load_documents()

    # Split the documents
    chunks = split_documents(documents)

    # Clear the database
    if args.clear:
        clear_database()

    # Update the database
    update_database(chunks)


if __name__ == "__main__":
    main()
