"""
Step 2: Basic RAG Pipeline
This script demonstrates the core RAG concept:
1. Load documents
2. Split into chunks
3. Create embeddings and store in vector database
4. Retrieve relevant chunks for a query
5. Generate answer using LLM

Run after successfully completing 01_test_ollama.py
"""

import os
from pathlib import Path

# LangChain imports
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
MODEL_NAME = "llama3.2"  # Change if using different model


def load_documents():
    """Load all text files from the data directory."""
    print(f"\n📁 Loading documents from: {DATA_DIR}")
    
    loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    
    documents = loader.load()
    print(f"✓ Loaded {len(documents)} documents")
    
    for doc in documents:
        filename = Path(doc.metadata["source"]).name
        print(f"  - {filename}")
    
    return documents


def split_documents(documents):
    """Split documents into smaller chunks for better retrieval."""
    print("\n✂️ Splitting documents into chunks...")
    
    # RecursiveCharacterTextSplitter is smart about where to split
    # It tries to keep paragraphs, sentences together
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # Characters per chunk
        chunk_overlap=50,     # Overlap between chunks (helps context)
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]  # Priority order for splits
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✓ Created {len(chunks)} chunks")
    
    # Show a sample chunk
    if chunks:
        print(f"\nSample chunk (first 200 chars):")
        print(f"'{chunks[0].page_content[:200]}...'")
    
    return chunks


def create_vector_store(chunks):
    """Create embeddings and store in Chroma vector database."""
    print("\n🧮 Creating embeddings and vector store...")
    print("(This may take a minute on first run)")
    
    # Use Ollama for embeddings too - keeps everything local
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    
    # Create the vector store
    # Chroma persists to disk so you don't rebuild every time
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    
    print(f"✓ Vector store created at: {CHROMA_DIR}")
    return vector_store


def create_qa_chain(vector_store):
    """Create the question-answering chain."""
    print("\n🔗 Creating QA chain...")
    
    # Initialize the LLM
    llm = Ollama(model=MODEL_NAME)
    
    # Create a retriever from the vector store
    # k=3 means retrieve top 3 most relevant chunks
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Custom prompt template for IT support context
    prompt_template = """You are an IT support assistant. Use the following context to answer the question. 
If you cannot find the answer in the context, say "I don't have information about that in my knowledge base."

Context:
{context}

Question: {question}

Answer: """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create the chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" = put all retrieved docs into prompt
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    print("✓ QA chain ready")
    return qa_chain


def ask_question(qa_chain, question):
    """Ask a question and display the answer with sources."""
    print(f"\n❓ Question: {question}")
    print("-" * 50)
    
    result = qa_chain.invoke({"query": question})
    
    print(f"\n💬 Answer:\n{result['result']}")
    
    print(f"\n📚 Sources:")
    for doc in result['source_documents']:
        source = Path(doc.metadata['source']).name
        preview = doc.page_content[:100].replace('\n', ' ')
        print(f"  - {source}: '{preview}...'")


def main():
    print("=" * 60)
    print("SupportBot - Basic RAG Pipeline")
    print("=" * 60)
    
    # Step 1: Load documents
    documents = load_documents()
    
    # Step 2: Split into chunks
    chunks = split_documents(documents)
    
    # Step 3: Create vector store
    vector_store = create_vector_store(chunks)
    
    # Step 4: Create QA chain
    qa_chain = create_qa_chain(vector_store)
    
    # Step 5: Test with sample questions
    print("\n" + "=" * 60)
    print("Testing with sample questions")
    print("=" * 60)
    
    test_questions = [
        "How do I fix VPN disconnection issues?",
        "My printer is showing offline, what should I do?",
        "How do I reset my password?",
    ]
    
    for question in test_questions:
        ask_question(qa_chain, question)
        print("\n")
    
    print("=" * 60)
    print("SUCCESS! Basic RAG pipeline is working.")
    print("Next step: Run 03_chat_interface.py for interactive chat")
    print("=" * 60)


if __name__ == "__main__":
    main()
