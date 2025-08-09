import os
import getpass
from typing import Annotated

from functools import partial
from dotenv import load_dotenv

from IPython.display import Image, display
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import List, TypedDict

load_dotenv()


class ConversationManager:
    def __init__(self):
        self.vector_store = None
        self.embeddings = OpenAIEmbeddings()
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        print("ConversationManager initialized...")

    @tool
    def add_document_from_url(self, url: str) -> str:
        """Scrapes the content from the given URL and adds it to the knowledge base.
        Use this tool when a user provides a URL to a document that can be read."""
        print("=== TOOL: ADD DOCUMENT FROM URL ===")
        try:
            loader = WebBaseLoader(web_paths=[url])
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                add_start_index=True,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            )
            splitted_docs = text_splitter.split_documents(docs)

            if self.vector_store is None:
                print("Creating new vector store...")
                self.vector_store = FAISS.from_documents(
                    documents=splitted_docs, embedding=self.embeddings
                )
            else:
                print("Adding to existing vector store...")
                self.vector_store.add_documents(splitted_docs)

            return f"Successfully added document from {url} to the knowledge base."
        except Exception as e:
            return f"Error loading document from {url}: {e}"

    @tool
    def query_knowledge_base(self, question: str) -> str:
        """Answer the user's question based on the knowledge base.
        Use this tool when the user asks a question that isn't a URL."""
        print("=== TOOL: QUERY KNOWLEDGE BASE ===")
        if self.vector_store is None:
            return "No documents found in knowledge base. Please add a document first."

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

        prompt = ChatPromptTemplate.from_template(
            "Answer the user's question based on the following context:\n\n<context>{context}</context>\n\nQuestion: {question}"
        )

        document_chain = create_stuff_documents_chain(self.llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke({"question": question})
        return response["answer"]


def main():
    print("Hello from python-rag!")
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key: ")
    print("OpenAI API Key set.")

    manager = ConversationManager()
    tools = [manager.add_document_from_url, manager.query_knowledge_base]

    memory = MemorySaver()
    agent_executor = create_react_agent(manager.llm, tools, checkpointer=memory)
    print("ReAct agent with memory created successfully.")

    # Run the conversational agent
    config = {"configurable": {"thread_id": "user_convo_2"}}
    print(
        "\n=== Agent is ready. You can now provide URLs to add the knowledge base or ask questions about them. ==="
    )
    print("Type 'exit' to end the conversation! ===\n")

    while True:
        user_input = input("\n\nYou: ")
        if user_input.lower() in ("exit", "quit"):
            print("Agent: Goodbye!")
            break

        events = agent_executor.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config,
            stream_mode="values",
        )
        for event in events:
            if "messages" in event:
                event["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()
