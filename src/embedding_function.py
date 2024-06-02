import os

from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_openai import OpenAIEmbeddings

HF_INFERENCE_API_KEY = os.environ["HF_INFERENCE_API_KEY"]


def get_huggingface_embedding_function(model_name):
    """
    Get the HuggingFace embedding function.
    """
    # Create the HuggingFace embedding function
    return HuggingFaceInferenceAPIEmbeddings(api_key=HF_INFERENCE_API_KEY, model_name=model_name)


def get_gpt4all_embedding_function():
    """
    Get the GPT4All embedding function.
    """
    # Create the GPT4All embedding function
    return GPT4AllEmbeddings()


def get_openai_embedding_function():
    """
    Get the OpenAI embedding function.
    """
    # Create the OpenAI embedding function
    return OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])


def get_embedding_bedrock_function():
    embeddings = BedrockEmbeddings(
        credentials_profile_name="default", region_name="us-east-1"
    )
    return embeddings
