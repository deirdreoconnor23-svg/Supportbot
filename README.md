# Patch  - IT Knowledge Assistant

A RAG-powered IT support knowledge base that keeps all data local and private.

## What This Project Does

Patch takes your IT documentation (troubleshooting guides, runbooks, past ticket solutions) and makes them searchable using natural language. Ask a question, get an answer with sources - all running locally on your machine.

## Demo

▶️ [Watch the demo video](https://youtu.be/TQzZCyZ6V50)


## Project Structure

```
supportbot/
├── data/                    # Your IT documentation goes here
│   ├── vpn_troubleshooting.txt
│   ├── windows11_common_issues.txt
│   ├── printer_setup.txt
│   └── password_reset.txt
├── src/
│   ├── 01_test_ollama.py   # Step 1: Verify Ollama works
│   ├── 02_basic_rag.py     # Step 2: Simple RAG pipeline
│   └── 03_chat_interface.py # Step 3: Interactive chat
├── requirements.txt
└── README.md
```

## Setup Instructions

### Step 1: Install Ollama

1. Download from https://ollama.com
2. Run the installer
3. Open terminal and run:
   ```bash
   ollama run llama3.2
   ```
4. Wait for model to download (~2GB)
5. Test with a question, then type `/bye` to exit

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

### Step 3: Run the Test Scripts

```bash
# First, verify Ollama connection
python src/01_test_ollama.py

# Then try the basic RAG pipeline
python src/02_basic_rag.py

# Finally, interactive chat
python src/03_chat_interface.py
```

## Adding Your Own Documents

Drop `.txt` or `.pdf` files into the `data/` folder. The system will automatically:
1. Load and chunk them
2. Create embeddings
3. Store in the vector database
4. Make them searchable

## Hardware Requirements

- Minimum: 8GB RAM (for 3B parameter models)
- Recommended: 16GB RAM (for 7B parameter models)
- Storage: ~5GB for models + your documents

## Privacy Note

All processing happens locally. Your documents never leave your machine.
No API keys required. No cloud services. No data leakage.

## Next Steps (Week 2+)

- [ ] Add PDF support
- [ ] Implement conversation memory
- [ ] Add source citations with page numbers
- [ ] Add confidence scoring

## Troubleshooting

**Ollama not responding?**
- Check if Ollama is running: `ollama list`
- Restart: `ollama serve`

**Out of memory?**
- Try a smaller model: `ollama run phi3`
- Close other applications

**Slow responses?**
- Normal for first query (model loading)
- Subsequent queries should be faster
