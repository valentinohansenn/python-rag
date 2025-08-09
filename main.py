import os
import getpass
import faiss
import pickle

from functools import partial, update_wrapper
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool, Tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

# from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
os.environ.setdefault("USER_AGENT", "python-rag/0.1 (contact: you@example.com)")


class AddUrlArgs(BaseModel):
    url: str = Field(
        description="The URL of the document to add to the knowledge base."
    )


class AddFileArgs(BaseModel):
    file_path: str = Field(description="The file path of the document to add.")


class QueryKBArgs(BaseModel):
    question: str = Field(
        description="The user's question to answer from the knowledge base."
    )


class ConversationManager:
    def __init__(self, thread_id: str = "user_convo_3"):
        self.vector_store = None
        self.embeddings = OpenAIEmbeddings()
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")

        self.index_path = f"./faiss_index_{thread_id}"
        self.pkl_path = f"./faiss_pkl_{thread_id}.pkl"
        print(f"ConversationManager initialized for thread {thread_id}...")

        if os.path.exists(self.index_path) and os.path.exists(self.pkl_path):
            print("Loading existing vector store...")
            try:
                index = faiss.read_index(self.index_path)
                with open(self.pkl_path, "rb") as f:
                    docstore, index_to_docstore_id = pickle.load(f)

                self.vector_store = FAISS(
                    embedding_function=self.embeddings,
                    index=index,
                    docstore=docstore,
                    index_to_docstore_id=index_to_docstore_id,
                )
                print("Existing vector store loaded successfully.")
            except Exception as e:
                print(f"Error loading existing vector store: {e}")

    def _save_knowledge_base(self):
        if self.vector_store:
            print("Saving vector store...")
            faiss.write_index(self.vector_store.index, self.index_path)
            with open(self.pkl_path, "wb") as f:
                pickle.dump(
                    [
                        self.vector_store.docstore,
                        self.vector_store.index_to_docstore_id,
                    ],
                    f,
                )
            print("Vector store saved successfully.")

    # @tool(args_schema=AddUrlArgs)
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

            self._save_knowledge_base()
            return f"Successfully added document from {url} to the knowledge base."
        except Exception as e:
            return f"Error loading document from {url}: {e}"

    # @tool(args_schema=AddFileArgs)
    def add_document_from_filepath(self, file_path: str) -> str:
        """Reads a document from the given file path and adds it to the knowledge base.
        Use this tool when a user provides a file path to a document that can be read.
        """
        print("=== TOOL: ADD DOCUMENT FROM FILE PATH ===")
        try:
            if not os.path.exists(file_path):
                return f"Error: File not found: {file_path}"

            if file_path.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                return f"Error: Unsupported file type: {file_path}"

            docs = loader.load_and_split(
                text_splitter=RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    add_start_index=True,
                    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                )
            )

            if self.vector_store is None:
                print("Creating new vector store...")
                self.vector_store = FAISS.from_documents(
                    documents=docs, embedding=self.embeddings
                )
            else:
                print("Adding to existing vector store...")
                self.vector_store.add_documents(docs)

            self._save_knowledge_base()
            return f"Successfully added document from {os.path.basename(file_path)} to the knowledge base."

        except Exception as e:
            return f"Error loading document from {file_path}: {e}"

    # @tool(args_schema=QueryKBArgs)
    def query_knowledge_base(self, question: str) -> str:
        """Answer the user's question based on the knowledge base."""
        print("=== TOOL: QUERY KNOWLEDGE BASE ===")
        if self.vector_store is None:
            return "No documents found in knowledge base. Please add a document first."

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

        prompt = ChatPromptTemplate.from_template(
            "Answer the user's question based on the following context:\n\n<context>{context}</context>\n\nQuestion: {question}"
        )

        document_chain = create_stuff_documents_chain(self.llm, prompt)

        retrieval_chain = {
            "context": retriever,
            "question": RunnablePassthrough(),
        } | document_chain

        response = retrieval_chain.invoke(question)
        answer = response

        suggestion_prompt = ChatPromptTemplate.from_template(
            """Given the original question and the answer provided, please generate 3 insightful and relevant follow-up questions that the user might want to ask next to dive deeper into the topic.

            Original Question: {question}
            Answer: {answer}

            Suggest 3 follow-up questions as a bulleted list:"""
        )

        suggestion_chain = suggestion_prompt | self.llm | StrOutputParser()
        followup_questions = suggestion_chain.invoke(
            {"question": question, "answer": answer}
        )

        return f"{answer}\n\nHere are some follow-up questions you might find helpful:\n{followup_questions}"


def main():
    print("Hello from python-rag!")
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key: ")

    if not os.environ.get("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API Key: ")

    config = {"configurable": {"thread_id": "user_convo_3"}}
    manager = ConversationManager(thread_id="user_convo_3")

    web_search_tool = TavilySearch(max_results=3)
    web_search_tool.description = "A search engine useful for answering questions about new or current events, or for general knowledge questions. Use this when the user asks a question that likely cannot be answered by the documents in the knowledge base."

    tools = [
        Tool(
            name="add_document_from_url",
            description="Scrapes the content from the given URL and adds it to the knowledge base.",
            func=lambda url: manager.add_document_from_url(url=url),
            args_schema=AddUrlArgs,  # Use the Pydantic schema you already created
        ),
        Tool(
            name="add_document_from_filepath",
            description="Reads a document from the given file path and adds it to the knowledge base.",
            func=lambda file_path: manager.add_document_from_filepath(
                file_path=file_path
            ),
            args_schema=AddFileArgs,
        ),
        Tool(
            name="query_knowledge_base",
            description="Answer the user's question based on the knowledge base.",
            func=lambda question: manager.query_knowledge_base(question=question),
            args_schema=QueryKBArgs,
        ),
        web_search_tool,
    ]

    memory = MemorySaver()
    agent_executor = create_react_agent(manager.llm, tools, checkpointer=memory)
    print("ReAct agent with memory created successfully.")

    # Run the conversational agent
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
