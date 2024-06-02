from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from embedding_function import *

DB_PATH = "../MATH_DB"

PROMPT_TEMPLATE = """
Answer the question based on the following context if relevant:

<context>
{context}
</context>

If the context is not relevant, provide the best answer you can.

---

"""


def query_rag(query_text: str):
    # Prepare the DB.
    model_name = "BAAI/bge-m3"
    embedding_function = get_huggingface_embedding_function(model_name)
    db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                PROMPT_TEMPLATE,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    print("Context Text: ", context_text)

    # Point to the local server
    llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    document_chain = create_stuff_documents_chain(llm, prompt)
    context_doc = [doc for doc, _score in results]

    response_text = document_chain.invoke(
        {
            "context": context_doc,
            "messages": [
                HumanMessage(content=query_text)
            ],
            "question": query_text
        }
    )

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


def main():
    while True:
        query_text = input("Enter your query (or 'q' to quit): ")
        if query_text.lower() == 'q':
            break
        elif query_text != "":
            query_rag(query_text)


if __name__ == "__main__":
    main()
