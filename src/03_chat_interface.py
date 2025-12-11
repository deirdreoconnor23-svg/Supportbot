"""
Step 3: Interactive Chat Interface
A command-line chat interface for SupportBot.
Run after 02_basic_rag.py has created the vector store.
"""

import os
from pathlib import Path

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# Configuration
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
MODEL_NAME = "llama3.2"


def load_existing_vectorstore():
    """Load the vector store created by 02_basic_rag.py"""
    
    if not CHROMA_DIR.exists():
        print("❌ Error: Vector store not found!")
        print("Please run 02_basic_rag.py first to create the knowledge base.")
        return None
    
    print("📂 Loading existing knowledge base...")
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    
    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )
    
    # Check if there's actually data
    collection = vector_store.get()
    doc_count = len(collection['ids'])
    print(f"✓ Loaded {doc_count} document chunks")
    
    return vector_store


def create_chat_chain(vector_store):
    """Create a conversational chain."""
    
    llm = Ollama(model=MODEL_NAME)
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # IT support focused prompt
    prompt_template = """You are SupportBot, a helpful IT support assistant. 
Use the following context from the knowledge base to answer the user's question.

Guidelines:
- Be concise and practical
- If the answer is in the context, provide step-by-step instructions where appropriate
- If you cannot find relevant information, say so clearly
- Always be helpful and professional

Context from knowledge base:
{context}

User Question: {question}

SupportBot: """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain


def print_welcome():
    """Display welcome message."""
    print("\n" + "=" * 60)
    print("🤖 SupportBot - IT Knowledge Assistant")
    print("=" * 60)
    print("\nAsk me anything about:")
    print("  • VPN troubleshooting")
    print("  • Windows 11 issues")
    print("  • Printer setup")
    print("  • Password resets")
    print("\nCommands:")
    print("  'quit' or 'exit' - End the session")
    print("  'sources'        - Show/hide source documents")
    print("=" * 60 + "\n")


def chat_loop(qa_chain):
    """Main chat interaction loop."""
    
    show_sources = False
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Handle commands
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("\n👋 Thanks for using SupportBot. Goodbye!")
                break
            
            if user_input.lower() == 'sources':
                show_sources = not show_sources
                status = "ON" if show_sources else "OFF"
                print(f"📚 Source display is now {status}\n")
                continue
            
            # Process the question
            print("\n🔍 Searching knowledge base...")
            
            result = qa_chain.invoke({"query": user_input})
            
            # Display answer
            print(f"\n🤖 SupportBot: {result['result']}")
            
            # Optionally show sources
            if show_sources and result.get('source_documents'):
                print("\n📚 Sources used:")
                for i, doc in enumerate(result['source_documents'], 1):
                    source = Path(doc.metadata['source']).name
                    print(f"  {i}. {source}")
            
            print()  # Empty line for readability
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.\n")


def main():
    # Load existing vector store
    vector_store = load_existing_vectorstore()
    
    if vector_store is None:
        return
    
    # Create chat chain
    print("🔗 Initializing chat system...")
    qa_chain = create_chat_chain(vector_store)
    print("✓ Ready!\n")
    
    # Display welcome and start chat
    print_welcome()
    chat_loop(qa_chain)


if __name__ == "__main__":
    main()
