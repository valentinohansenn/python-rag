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
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
)

from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import List, TypedDict

load_dotenv()


def set_openai_api_key():
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key: ")
    print("OpenAI API Key set.")

# 1. Define the State for the Graph
class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]

# 2. Define the tool
def setup_retriever():
    print("=== NODE: SETUP RETRIEVER ===")
    loader = WebBaseLoader(web_paths=["https://lilianweng.github.io/posts/2023-06-23-agent/"])
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    splitted_docs = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(documents=splitted_docs, embedding=OpenAIEmbeddings())
    return vectorstore.as_retriever(search_kwargs={"k": 4})

retriever = setup_retriever()

@tool
def retrieve_documents(query: str) -> List[Document]:
    """Looks up relevant documents from a knowledge base based on the user's query."""
    print("=== TOOL: RETRIEVE ===")
    retrieved_docs = retriever.invoke(query)
    return [doc.__dict__ for doc in retrieved_docs]

# 3. Define the nodes - the "brain" of the agent
def call_model(state: MessagesState, llm_with_tools):
    print("=== NODE: CALL MODEL ===")
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def call_tool(state: MessagesState):
    last_message = state["messages"][-1]

    tool_call = last_message.tool_calls[0]
    if tool_call["name"] == "retrieve_documents":
        tool_output = retrieve_documents.invoke(tool_call["args"])
        return {"messages": [ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])]}
    return state

# Conditional edge decides whether to continue the loop or end the graph
def should_continue(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return END
    return "call_tool"

def main():
    print("Hello from python-rag!")
    set_openai_api_key()
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")

    # Create a new llm with the tool bound to it
    llm_with_tools = llm.bind_tools([retrieve_documents])

    # Create a new StateGraph with our defined state
    workflow = StateGraph(MessagesState)

    # Use partial to "bake" the llm_with_tools into our node function
    model_node = partial(call_model, llm_with_tools=llm_with_tools)

    # Add nodes to the graph
    workflow.add_node("agent", model_node)
    workflow.add_node("call_tool", call_tool)

    # Define the flow of the graph
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {
        "call_tool": "call_tool",
        END: END,
    })
    workflow.add_edge("call_tool", "agent")

    app = workflow.compile()
    print("Compiled workflow.")

    display(Image(app.get_graph().draw_mermaid_png()))

    query = input("\nEnter your query: ")
    print("\n === Executing workflow === ")
    final_state = app.invoke({"messages": [HumanMessage(content=query)]})
    print("\n === Workflow completed === ")

    final_answer = final_state["messages"][-1]
    print("Final answer: ", final_answer.content)

    print("\n === FULL MESSAGE HISTORY ===")
    for message in final_state["messages"]:
        print(f"**{message.type.upper()}**:\n{message.content}\n")


if __name__ == "__main__":
    main()
