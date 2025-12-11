"""
Step 1: Test Ollama Connection
Run this first to verify Ollama is installed and working.

Before running:
1. Install Ollama from https://ollama.com
2. Run: ollama run llama3.2 (to download the model)
3. Type /bye to exit
4. Then run this script
"""

from langchain_community.llms import Ollama

def test_ollama():
    print("=" * 50)
    print("Testing Ollama Connection")
    print("=" * 50)
    
    # Try to connect to Ollama
    try:
        # Initialize the Ollama model
        # Change model name if you're using a different one
        llm = Ollama(model="llama3.2")
        print("✓ Connected to Ollama successfully")
        
    except Exception as e:
        print(f"✗ Failed to connect to Ollama: {e}")
        print("\nTroubleshooting:")
        print("1. Is Ollama installed? Download from https://ollama.com")
        print("2. Is Ollama running? Open terminal and run: ollama serve")
        print("3. Is the model downloaded? Run: ollama run llama3.2")
        return False
    
    # Test a simple query
    print("\nSending test query...")
    try:
        response = llm.invoke("What is 2 + 2? Reply with just the number.")
        print(f"✓ Response received: {response.strip()}")
        
    except Exception as e:
        print(f"✗ Query failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("SUCCESS! Ollama is working correctly.")
    print("You're ready for Step 2: basic_rag.py")
    print("=" * 50)
    return True


if __name__ == "__main__":
    test_ollama()
