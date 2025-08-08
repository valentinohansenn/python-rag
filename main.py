import os
import getpass
import bs4

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def set_openai_api_key():
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key: ")
    print("OpenAI API Key set.")


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


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def main():
    print("Hello from python-rag!")
    set_openai_api_key()
    llm = init_llm()

    docs = load_docs()
    splitted_docs = split_docs(docs)

    db = add_documents(documents=splitted_docs)

    retriever = db.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Use the context to answer the user. If unsure, say you don't know.\n\nContext:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    query = input("Enter your query: ")
    docs = chain.invoke({"question": query})
    print(docs)


if __name__ == "__main__":
    main()
