# Patch - IT Knowledge Assistant

A RAG-powered IT support knowledge base that keeps all data local and private.

## What This Project Does

Patch demonstrates how to build an AI-powered IT support assistant that:
- Runs entirely locally using Ollama (no cloud dependency)
- Uses a **two-model architecture**: dedicated embedding model for retrieval + LLM for generation
- Maintains **conversation context** across follow-up questions
- Uses ChromaDB vector database for document storage and similarity search
- Maintains complete data privacy with zero external API calls

This project showcases production-ready RAG patterns including proper embedding model selection, conversation history injection, and prompt engineering for accurate responses.

## Demo

[Watch the demo video](https://youtu.be/TQzZCyZ6V50)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUESTION                            │
│                  "My VPN keeps disconnecting"                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 CONTEXT INJECTION LAYER                         │
│         (Adds recent conversation history to query)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RETRIEVAL PHASE                              │
│  ┌─────────────────┐    ┌─────────────────────────────────┐    │
│  │ nomic-embed-text│───▶│      ChromaDB Vector Store      │    │
│  │ (embedding model)│    │  (similarity search for docs)   │    │
│  └─────────────────┘    └─────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GENERATION PHASE                             │
│  ┌─────────────────┐                                           │
│  │    llama3.2     │  + Retrieved docs + Engineered prompt     │
│  │ (generative LLM)│  = Accurate, documentation-backed answer  │
│  └─────────────────┘                                           │
└─────────────────────────────────────────────────────────────────┘
```

### Why Two Models?

| Model | Purpose | Why It Matters |
|-------|---------|----------------|
| **nomic-embed-text** | Converts text to vectors for semantic search | Dedicated embedding models outperform generative LLMs at similarity matching by 15x+ |
| **llama3.2** | Generates human-readable responses | Optimized for natural language generation and instruction-following |

## Key Features

### Conversation Context Preservation
Follow-up questions maintain topic relevance. Ask about email, then say "how do I fix that?" - the system remembers you were discussing email.

### Accurate Document Retrieval
Using a dedicated embedding model ensures queries like "VPN disconnecting" find VPN documentation, not unrelated topics.

### Exact Documentation Steps
Prompt engineering ensures the LLM reproduces specific steps from your documentation rather than generic advice.

## Project Structure

```
Supportbot/
├── data/                       # IT documentation (knowledge base)
│   ├── vpn_troubleshooting.txt
│   ├── email_troubleshooting.txt
│   ├── printer_troubleshooting.txt
│   ├── password_reset.txt
│   └── ...
├── src/
│   ├── patch-app.py            # Streamlit web interface
│   ├── patch_api.py            # FastAPI REST backend
│   ├── 01_test_ollama.py       # Ollama connection test
│   ├── 02_basic_rag.py         # Basic RAG pipeline
│   └── 03_chat_interface.py    # CLI chat interface
├── chroma_db/                  # Vector database storage
├── DEBUGGING_LOG.md            # Technical details of RAG improvements
├── LEARNING_NOTES.md           # Beginner-friendly RAG concepts guide
├── requirements.txt
└── README.md
```

## Setup Instructions

### Step 1: Install Ollama

1. Download from https://ollama.com
2. Run the installer
3. Pull both required models:
   ```bash
   # Generative model for responses
   ollama pull llama3.2

   # Embedding model for semantic search (critical for retrieval quality)
   ollama pull nomic-embed-text
   ```

### Step 2: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Build the Knowledge Base

```bash
# Run the basic RAG script to build the vector database
python src/02_basic_rag.py
```

### Step 4: Run the Application

**Option A: Streamlit Web Interface**
```bash
streamlit run src/patch-app.py
```

**Option B: FastAPI Backend**
```bash
python src/patch_api.py
# API available at http://127.0.0.1:8000
# Interactive docs at http://127.0.0.1:8000/docs
```

**Option C: CLI Chat**
```bash
python src/03_chat_interface.py
```

## API Usage

The FastAPI backend supports conversation history for context-aware responses:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I fix that?",
    "conversation_history": [
      {"role": "user", "content": "My email wont sync"},
      {"role": "assistant", "content": "Try restarting Outlook..."}
    ]
  }'
```

## Adding Your Own Documents

Drop `.txt` files into the `data/` folder, then rebuild the knowledge base:

1. Add your documents to `data/`
2. Delete the old database: `rm -rf chroma_db/`
3. Rebuild: `python src/02_basic_rag.py`

The system will automatically chunk documents (1200 chars with 200 overlap) and create embeddings.

## Technical Documentation

| Document | Description |
|----------|-------------|
| [DEBUGGING_LOG.md](DEBUGGING_LOG.md) | Technical chronicle of issues diagnosed and fixed, including retrieval accuracy improvements and context injection implementation |
| [LEARNING_NOTES.md](LEARNING_NOTES.md) | Beginner-friendly guide to RAG concepts with analogies - useful for explaining to employers or non-technical stakeholders |

## Hardware Requirements

- Minimum: 8GB RAM
- Recommended: 16GB RAM
- Storage: ~5GB for models + your documents

## Privacy

All processing happens locally. Your documents never leave your machine.
No API keys required. No cloud services. No data leakage.

## Troubleshooting

**Ollama not responding?**
```bash
ollama list    # Check if models are available
ollama serve   # Restart Ollama service
```

**Poor search results?**
- Ensure you pulled `nomic-embed-text`: `ollama pull nomic-embed-text`
- Rebuild the database after pulling: `rm -rf chroma_db/ && python src/02_basic_rag.py`

**Context not maintained?**
- Ensure you're using the latest `patch-app.py` or `patch_api.py`
- For API usage, pass `conversation_history` parameter

**Out of memory?**
- Try a smaller generative model: `ollama pull phi3`
- Update `MODEL_NAME` in the source files

## License

MIT
