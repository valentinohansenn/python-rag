import os
import getpass
import bs4

from functools import partial
from dotenv import load_dotenv

from IPython.display import Image, display
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from langgraph.graph import StateGraph, END
from typing_extensions import List, TypedDict

load_dotenv()


def set_openai_api_key():
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key: ")
    print("OpenAI API Key set.")


class State(TypedDict):
    query: str
    context: List[Document]
    response: str


def retrieve(state: State, retriever):
    print("=== NODE: RETRIEVE ===")
    query = state["query"]
    context = retriever.invoke(query)
    return {"context": context}


def generate(state: State, llm):
    print("=== NODE: GENERATE ===")
    query = state["query"]
    context = state["context"]

    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": format_docs(context), "question": query})
    return {"response": response}


def init_llm():
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    return llm


def load_docs(web_path="https://lilianweng.github.io/posts/2023-06-23-agent/"):
    bs4_strainer = bs4.SoupStrainer(
        class_=("post-title", "post-header", "post-content")
    )
    loader = WebBaseLoader(
        web_paths=[web_path],
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    total_chars = sum(len(doc.page_content) for doc in docs)
    print(f"Loaded {len(docs)} documents with total characters: {total_chars}")
    return docs


def split_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )

    all_splits = text_splitter.split_documents(docs)
    print(f"Split the given document into {len(all_splits)} sub-documents.")
    return all_splits


def add_documents(docs):
    db = FAISS.from_documents(
        docs, OpenAIEmbeddings(model="text-embedding-3-large"), normalize_L2=True
    )
    print(f"Indexed {len(db.docstore._dict)} documents.")

    return db


def main():
    print("Hello from python-rag!")
    set_openai_api_key()
    llm = init_llm()

    docs = load_docs()
    splitted_docs = split_docs(docs)

    db = add_documents(docs=splitted_docs)

    retriever = db.as_retriever(search_kwargs={"k": 4})
    print("Created retriever.")

    # Create partials for our nodes to "bake" the retriever and llm into the nodes
    retrieve_node = partial(retrieve, retriever=retriever)
    generate_node = partial(generate, llm=llm)

    # Create a new StateGraph with our defined state
    workflow = StateGraph(State)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    # Define the flow of the graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    app = workflow.compile()
    print("Compiled workflow.")

    display(Image(app.get_graph().draw_mermaid_png()))

    query = input("\nEnter your query: ")
    print("\n === Executing workflow === ")
    final_states = app.invoke({"query": query})

    print("\n === Workflow complete === ")
    print(f"Context: {final_states['context']}\n\n")
    print(f"Response: {final_states['response']}")

    # For streaming the response
    # for message, metadata in graph.stream({"query": query}, stream_mode="messages"):
    #     print(message, end="|", flush=True)


if __name__ == "__main__":
    main()
