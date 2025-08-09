## python-rag

A simple but capable Retrieval-Augmented Generation (RAG) console agent built with LangChain and FAISS. It lets you:

- Add documents from web URLs or local PDF files to a persistent FAISS vector store
- Ask questions that are answered using your knowledge base
- Use a ReAct agent with tool-use and short-term memory across a conversation
- Fall back to web search (Tavily) for questions not covered by your KB

The app persists its vector store to disk so your knowledge base survives restarts.


### Key features

- **RAG over your docs**: Chunking with `RecursiveCharacterTextSplitter`, embeddings via `OpenAIEmbeddings`, fast similarity search with `FAISS`.
- **ReAct agent + tools**: Natural-language commands trigger tools to add docs and query the KB. Includes a web search tool via Tavily.
- **Persistent knowledge base**: FAISS index and metadata are saved locally per conversation thread ID.
- **Follow-up suggestions**: After answering, the agent proposes follow-up questions to explore deeper.


## How it works

- `ConversationManager` manages the vector store and the core tools:
  - `add_document_from_url(url)` – scrape and ingest a web page
  - `add_document_from_filepath(file_path)` – ingest a local PDF
  - `query_knowledge_base(question)` – retrieve relevant chunks and answer via the LLM
- A ReAct agent (`langgraph.prebuilt.create_react_agent`) wires these tools together with an LLM (`gpt-4o-mini` via OpenAI) and a short-term memory checkpointer (`MemorySaver`).
- The knowledge base persists in the project root by default as:
  - `faiss_index_user_convo_3`
  - `faiss_pkl_user_convo_3.pkl`
- The default conversation/thread ID is `user_convo_3`. Change this if you want separate KBs per conversation.


## Requirements

- Python 3.11+
- OpenAI API key
- Tavily API key

The code also uses `python-dotenv` to optionally load environment variables from a `.env` file.


## Installation

You can use either uv (recommended) or pip + venv.

### Option A: uv (recommended)

1) Install uv
- macOS:
  - Homebrew: `brew install uv`
  - Or via official installer: `curl -LsSf https://astral.sh/uv/install.sh | sh`

2) Create the environment and install dependencies

```bash
uv sync
```

3) Run the app

```bash
uv run python main.py
```


### Option B: pip + venv

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install .
python main.py
```


## Configuration

Set your API keys via environment variables or a `.env` file in the project root.

### Required environment variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `TAVILY_API_KEY`: Your Tavily API key

If these are not set, the app will securely prompt you on startup.

### Optional environment variables

- `USER_AGENT`: Defaults to `python-rag/0.1 (contact: you@example.com)`

### Example .env

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
USER_AGENT=python-rag/0.1 (contact: me@mydomain.com)
```


## Usage

Start the agent:

```bash
python main.py
```

You will see prompts indicating the agent is ready. Type natural-language requests; the agent will decide when to use tools.

### Example interactions

- **Add a URL to the KB**
  - "Add `https://langchain.readthedocs.io/` to the knowledge base"
  - "Ingest `https://example.com/blog/post`"

- **Add a local PDF to the KB**
  - "Add the file `/Users/me/Documents/paper.pdf`"
  - "Ingest `/absolute/path/to/report.pdf`"

- **Ask questions from your KB**
  - "What are the key takeaways from the document we just added?"
  - "Summarize the methodology section and list three limitations."

- **General knowledge / current events**
  - "What are the latest best practices for vector databases?"

The agent will retrieve relevant context and answer. It will also suggest follow-up questions.


## Data and persistence

- Vector index and metadata are persisted in files named based on the thread ID (default: `user_convo_3`).
- To keep separate KBs per conversation, change the thread ID in `main.py`:
  - `config = {"configurable": {"thread_id": "your_thread"}}`
  - `ConversationManager(thread_id="your_thread")`

Each unique `thread_id` yields separate `faiss_index_<thread_id>` and `faiss_pkl_<thread_id>.pkl` files.


## Customization

- **Model**: The default LLM is `gpt-4o-mini`. Update the call in `main.py`:
  - `init_chat_model("gpt-4o-mini", model_provider="openai")`
- **Chunking**: Adjust `chunk_size`, `chunk_overlap`, or `separators` in the text splitter.
- **Retriever**: Tune `search_kwargs={"k": 4}` in `as_retriever`.
- **File types**: Currently, the local file ingestion tool supports PDFs via `PyPDFLoader`. Extend this for other formats as needed.


## Troubleshooting

- "No documents found in knowledge base": Add at least one URL/PDF before asking KB questions.
- FAISS install issues on macOS:
  - Ensure Python 3.11 and a recent `pip`/`setuptools`/`wheel`.
  - If you use Apple Silicon, prefer `uv` or a recent Python from `pyenv`/Homebrew.
- Web loader errors: Some sites block scraping or require JS rendering beyond simple loaders.
- API errors: Verify `OPENAI_API_KEY` and `TAVILY_API_KEY` are set and valid.


## Tech stack

- LangChain (Community/Core, Text Splitters)
- LangGraph (ReAct agent, memory)
- FAISS (vector store)
- OpenAI (embeddings + chat model)
- Tavily (web search tool)
- python-dotenv, pydantic, PyPDF


## License

Add your preferred license here.
