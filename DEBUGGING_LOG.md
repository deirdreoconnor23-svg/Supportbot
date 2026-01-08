# Debugging Log: RAG Chatbot Improvements

**Date:** January 2025
**Project:** Patch - IT Support Knowledge Assistant
**Issues Fixed:** Context loss between messages, generic responses ignoring documentation

---

## Issue 1: Conversation Context Loss

### Symptom
User asks about email, then follows up with "how do I fix that?", and the chatbot responds about printers instead of email.

### Root Cause
Neither `patch-app.py` nor `patch_api.py` passed conversation history to the RAG chain. Each query was stateless.

**In patch-app.py (line 489):**
```python
result = qa_chain.invoke({"query": user_input})  # Only current message sent
```

The `st.session_state.messages` stored chat history for display only - never sent to the LLM.

### Fix Applied
Added `build_enhanced_query()` function to inject recent conversation context:

```python
def build_enhanced_query(messages, current_question):
    recent_messages = messages[-(5):-1] if len(messages) > 1 else []
    if not recent_messages:
        return current_question

    context_parts = []
    for msg in recent_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        context_parts.append(f"{role}: {msg['content']}")

    context_str = "\n".join(context_parts)
    enhanced_query = f"""Previous conversation:
{context_str}

Current question: {current_question}"""
    return enhanced_query
```

**Files modified:**
- `src/patch-app.py` - Added function and updated query invocation
- `src/patch_api.py` - Added `MessageHistory` model, `conversation_history` parameter to API

---

## Issue 2: Generic Responses Ignoring Documentation

### Symptom
Query "My VPN keeps disconnecting" returned vague troubleshooting advice instead of the specific Device Manager steps documented in `vpn_troubleshooting.txt`.

### Investigation Steps

#### Step 1: Verified document content exists
```bash
cat data/vpn_troubleshooting.txt
```
Found detailed steps: Device Manager > Network adapters > Power Management > Uncheck "Allow computer to turn off this device"

#### Step 2: Checked retrieval settings
```
chunk_size: 500  (too small - splitting procedures across chunks)
chunk_overlap: 50
k: 5
```

#### Step 3: Examined prompt template
```
"Be brief and helpful" - caused summarization
"offer general advice" - gave LLM escape hatch
```

### Fix Part A: Chunk Size and Prompt

**Changed chunk settings:**
| Setting | Before | After |
|---------|--------|-------|
| chunk_size | 500 | 1200 |
| chunk_overlap | 50 | 200 |

**Updated prompt template:**
```python
prompt_template = """You are Patch, an IT support assistant.

RULES:
1. Use the EXACT steps and details from the reference material below - do not paraphrase or generalize
2. Include specific paths, commands, and settings mentioned in the reference material
3. If the reference material has numbered steps, preserve them in your answer
4. Focus only on the topic the user mentioned - don't mix in other topics
5. If the user indicates the issue is resolved, say "Great! Happy to be of service" and stop
6. Only if the reference material has NO relevant information, say "I don't have specific documentation for that issue" and offer to escalate
```

### Fix Part B: Embedding Model (Critical Fix)

After rebuilding, VPN queries still returned wrong documents. Direct similarity search revealed the problem:

```python
docs = vector_store.similarity_search_with_score('VPN disconnects', k=10)
# Result: VPN disconnection chunk NOT in top 10 results
# Top results: account_lockout.txt, printer_troubleshooting.txt
```

**Root Cause:** Using `llama3.2` (a generative LLM) for embeddings instead of a dedicated embedding model.

**Solution:** Switched to `nomic-embed-text`:
```bash
ollama pull nomic-embed-text
```

**Configuration change:**
```python
MODEL_NAME = "llama3.2"  # For LLM responses
EMBEDDING_MODEL = "nomic-embed-text"  # For vector embeddings

# Embeddings now use dedicated model
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# LLM still uses llama3.2
llm = Ollama(model=MODEL_NAME)
```

### Results After All Fixes

**Similarity search comparison:**

| Query: "VPN keeps disconnecting" | Before (llama3.2) | After (nomic-embed-text) |
|----------------------------------|-------------------|--------------------------|
| #1 Result | account_lockout.txt | vpn_troubleshooting.txt |
| Score (lower = better) | 5765 | 383 |
| Correct chunk found? | No (not in top 10) | Yes (#1 result) |

**Response quality:**

Before:
> "Can you tell me more about the problems you're experiencing? Are you getting an authentication error?"

After:
> "1. Open Device Manager (right-click Start > Device Manager)
> 2. Expand 'Network adapters'
> 3. Right-click your network adapter > Properties
> 4. Go to 'Power Management' tab
> 5. Uncheck 'Allow the computer to turn off this device to save power'
> 6. Click OK and restart"

---

## Important: Database Rebuild Required

ChromaDB doesn't auto-clear when rebuilding. Old embeddings persisted and contaminated results.

**Correct rebuild procedure:**
```bash
rm -rf chroma_db/  # Delete old database
python rebuild_script.py  # Create fresh with new embeddings
```

---

## Files Modified

| File | Changes |
|------|---------|
| `src/patch-app.py` | Added EMBEDDING_MODEL constant, build_enhanced_query(), updated chunk_size/overlap, new prompt |
| `src/patch_api.py` | Added EMBEDDING_MODEL constant, MessageHistory model, conversation_history parameter, updated chunk_size/overlap, new prompt |

---

## Key Takeaways

1. **Stateless RAG chains lose context** - Must explicitly inject conversation history
2. **Chunk size affects retrieval quality** - Too small splits related content
3. **Prompts control output behavior** - "Be brief" causes summarization; be explicit about using exact content
4. **Embedding model choice is critical** - Generative LLMs are not embedding models; use dedicated models like nomic-embed-text
5. **Vector databases accumulate** - Must delete old data before rebuilding with new embeddings
