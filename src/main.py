from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import gradio as gr

from embedding_function import *

DB_PATH = "../DATA_DB"

PROMPT_TEMPLATE = """
You are an AI assistant that provides detailed, accurate, and contextually relevant answers to user queries. Use the provided context to generate your response. If the context is not sufficient or does not directly relate to the query, use your general knowledge to answer as accurately as possible. Follow these guidelines:

1. Be concise but informative.
2. Use a professional and polite tone.
3. Cite any sources from the context where applicable.

Context:
<context>
{context}
</context>

Query: {query_text}

Response:
"""


def query_rag(query_text: str):
    # Prepare the DB.
    model_name = "BAAI/bge-m3"
    embedding_function = get_huggingface_embedding_function(model_name)
    db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)

    # Reduce the number of results fetched
    results = db.similarity_search_with_score(query_text, k=3)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                PROMPT_TEMPLATE,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # Point to the local server
    llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    document_chain = create_stuff_documents_chain(llm, prompt)
    context_doc = [doc for doc, _score in results]

    response_text = document_chain.invoke(
        {
            "context": context_doc,
            "query_text": query_text,
            "messages": [
                HumanMessage(content=query_text)
            ],

        }
    )

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response


def chat_interface(user_input, history):
    response = query_rag(user_input)
    history.append((user_input, response))
    return history, history


def main():
    # Define the Gradio interface
    with gr.Blocks() as interface:
        gr.Markdown("<h1 style='text-align: center;'>RAG Model Chat Interface</h1>")
        chatbot = gr.Chatbot()
        txt = gr.Textbox(show_label=False, placeholder="Enter your query...")

        txt.submit(chat_interface, inputs=[txt, chatbot], outputs=[chatbot, chatbot], queue=True)

    interface.launch()


if __name__ == "__main__":
    main()
