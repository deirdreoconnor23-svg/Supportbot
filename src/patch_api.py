"""
Patch API - FastAPI Backend
Extracts the RAG logic from Patch Streamlit app into a REST API.
This allows Patch to be called from any frontend (web, mobile, desktop).
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, List
import logging

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_NAME = "llama3.2"  # For LLM responses
EMBEDDING_MODEL = "nomic-embed-text"  # For vector embeddings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Patch API",
    description="IT Support Knowledge Assistant - Backend API",
    version="1.0.0"
)

# Global variable to cache the QA chain
qa_chain = None


# ---------- PYDANTIC MODELS (Data Validation) ----------

class MessageHistory(BaseModel):
    """Model for a single message in conversation history"""
    role: str  # "user" or "assistant"
    content: str


class QuestionRequest(BaseModel):
    """Request model for asking questions"""
    question: str
    conversation_history: Optional[List[MessageHistory]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "question": "How do I fix that?",
                "conversation_history": [
                    {"role": "user", "content": "My email won't sync"},
                    {"role": "assistant", "content": "Try restarting Outlook and checking your internet connection."}
                ]
            }
        }


class AnswerResponse(BaseModel):
    """Response model for answers"""
    answer: str
    sources: Optional[List[str]] = None
    

class StatusResponse(BaseModel):
    """Response model for system status"""
    status: str
    knowledge_base_ready: bool
    model_name: str
    document_count: Optional[int] = None


class StatsResponse(BaseModel):
    """Response model for statistics"""
    document_count: int
    chunk_count: Optional[int] = None


class RebuildResponse(BaseModel):
    """Response model for rebuild operation"""
    success: bool
    message: str
    documents_processed: int
    chunks_created: int


# ---------- RAG FUNCTIONS (From Your Streamlit App) ----------

def load_qa_chain():
    """
    Load or create the QA chain.
    This is the core RAG functionality from your Patch app.
    """
    global qa_chain
    
    if not CHROMA_DIR.exists():
        logger.warning("Chroma directory does not exist")
        return None
    
    try:
        logger.info("Loading embeddings and vector store...")
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vector_store = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings
        )
        
        logger.info("Initializing LLM...")
        llm = Ollama(model=MODEL_NAME)
        
        logger.info("Creating retriever...")
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        prompt_template = """You are Patch, an IT support assistant.

RULES:
1. Use the EXACT steps and details from the reference material below - do not paraphrase or generalize
2. Include specific paths, commands, and settings mentioned in the reference material
3. If the reference material has numbered steps, preserve them in your answer
4. Focus only on the topic the user mentioned - don't mix in other topics
5. If the user indicates the issue is resolved, say "Great! Happy to be of service" and stop
6. Only if the reference material has NO relevant information, say "I don't have specific documentation for that issue" and offer to escalate

Reference material:
{context}

User asked: {question}

Patch:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        logger.info("Creating QA chain...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        logger.info("QA chain loaded successfully!")
        return qa_chain
        
    except Exception as e:
        logger.error(f"Error loading QA chain: {e}")
        return None


def rebuild_knowledge_base():
    """
    Rebuild the vector store from documents in DATA_DIR.
    This is from your admin panel functionality.
    """
    try:
        logger.info(f"Loading documents from {DATA_DIR}...")
        loader = DirectoryLoader(
            str(DATA_DIR),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        documents = loader.load()
        
        if not documents:
            raise ValueError("No documents found in data directory")
        
        logger.info(f"Splitting {len(documents)} documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        logger.info(f"Creating embeddings for {len(chunks)} chunks...")
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        
        logger.info("Building vector store...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(CHROMA_DIR)
        )
        
        # Reload the QA chain with new data
        global qa_chain
        qa_chain = load_qa_chain()
        
        logger.info("Knowledge base rebuilt successfully!")
        return len(documents), len(chunks)
        
    except Exception as e:
        logger.error(f"Error rebuilding knowledge base: {e}")
        raise


def get_doc_count():
    """Count documents in the data directory"""
    if DATA_DIR.exists():
        return len(list(DATA_DIR.glob("*.txt")))
    return 0


def build_enhanced_query(conversation_history: Optional[List[MessageHistory]], current_question: str) -> str:
    """
    Build a query that includes recent conversation context.
    This helps the retriever find relevant documents for follow-up questions.
    """
    if not conversation_history:
        return current_question

    # Take the last 4 messages (2 exchanges) for context
    recent_messages = conversation_history[-4:]

    # Build context string from recent conversation
    context_parts = []
    for msg in recent_messages:
        role = "User" if msg.role == "user" else "Assistant"
        context_parts.append(f"{role}: {msg.content}")

    context_str = "\n".join(context_parts)

    # Create enhanced query with context
    enhanced_query = f"""Previous conversation:
{context_str}

Current question: {current_question}"""

    return enhanced_query


# ---------- API ENDPOINTS ----------

@app.on_event("startup")
async def startup_event():
    """Load the QA chain when the API starts"""
    global qa_chain
    logger.info("Starting Patch API...")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Chroma directory: {CHROMA_DIR}")
    
    qa_chain = load_qa_chain()
    
    if qa_chain:
        logger.info("✓ Patch API ready!")
    else:
        logger.warning("⚠ Knowledge base not found - will need to rebuild")


@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "name": "Patch API",
        "version": "1.0.0",
        "description": "IT Support Knowledge Assistant - Backend API",
        "endpoints": {
            "POST /ask": "Ask a question",
            "GET /status": "Check system status",
            "GET /stats": "Get statistics",
            "POST /rebuild": "Rebuild knowledge base",
            "GET /docs": "Interactive API documentation"
        }
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask Patch a question - Main RAG endpoint

    This is the core functionality from your Streamlit chat interface.
    Supports conversation history for context-aware follow-up questions.
    """
    global qa_chain

    if not qa_chain:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base not initialized. Please rebuild the knowledge base first."
        )

    try:
        logger.info(f"Processing question: {request.question}")

        # Build enhanced query with conversation context for better retrieval
        enhanced_query = build_enhanced_query(request.conversation_history, request.question)
        logger.info(f"Enhanced query built with {len(request.conversation_history) if request.conversation_history else 0} history messages")

        # Query the RAG system
        result = qa_chain.invoke({"query": enhanced_query})
        answer = result["result"]
        
        # Extract source document names
        sources = []
        if result.get("source_documents"):
            sources = [
                Path(doc.metadata.get("source", "unknown")).name 
                for doc in result["source_documents"]
            ]
            # Remove duplicates
            sources = list(dict.fromkeys(sources))
        
        logger.info(f"Answer generated successfully")
        
        return AnswerResponse(
            answer=answer,
            sources=sources if sources else None
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """
    Check if the system is ready
    """
    global qa_chain
    
    kb_ready = qa_chain is not None and CHROMA_DIR.exists()
    doc_count = get_doc_count() if DATA_DIR.exists() else 0
    
    return StatusResponse(
        status="ready" if kb_ready else "not_ready",
        knowledge_base_ready=kb_ready,
        model_name=MODEL_NAME,
        document_count=doc_count
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get statistics about the knowledge base
    """
    doc_count = get_doc_count()
    
    # Try to get chunk count from vector store
    chunk_count = None
    if CHROMA_DIR.exists():
        try:
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
            vector_store = Chroma(
                persist_directory=str(CHROMA_DIR),
                embedding_function=embeddings
            )
            collection = vector_store.get()
            chunk_count = len(collection['ids'])
        except Exception as e:
            logger.warning(f"Could not get chunk count: {e}")
    
    return StatsResponse(
        document_count=doc_count,
        chunk_count=chunk_count
    )


@app.post("/rebuild", response_model=RebuildResponse)
async def rebuild():
    """
    Rebuild the knowledge base from documents in the data directory
    
    This is the equivalent of your admin panel's "Rebuild Knowledge Base" button.
    """
    try:
        logger.info("Starting knowledge base rebuild...")
        doc_count, chunk_count = rebuild_knowledge_base()
        
        return RebuildResponse(
            success=True,
            message="Knowledge base rebuilt successfully",
            documents_processed=doc_count,
            chunks_created=chunk_count
        )
        
    except Exception as e:
        logger.error(f"Rebuild failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to rebuild knowledge base: {str(e)}"
        )


# ---------- OPTIONAL: HEALTH CHECK ----------

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    
    # Create directories if they don't exist
    DATA_DIR.mkdir(exist_ok=True)
    CHROMA_DIR.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("🟠 Starting Patch API Server")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")
    print(f"Chroma directory: {CHROMA_DIR}")
    print("=" * 60)
    
    # Run the server
    uvicorn.run(app, host="127.0.0.1", port=8000)
