"""
Patch - IT Support Knowledge Assistant
Clean, Squarespace-inspired interface connected to local RAG model.
"""

import streamlit as st
from pathlib import Path
import base64
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

# Page configuration
st.set_page_config(
    page_title="Patch – IT Support Assistant",
    page_icon="🟠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    .stApp {
        background: #FAFAFA;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    .block-container {
        max-width: 720px;
        padding-top: 48px;
        padding-bottom: 48px;
        padding-left: 24px;
        padding-right: 24px;
    }
    
    /* Header */
    .patch-header {
        display: flex;
        align-items: center;
        gap: 18px;
        margin-bottom: 32px;
        justify-content: center;
    }
    
    .patch-header > div:last-child {
        margin-top: -4px;
    }
    
    .patch-logo {
        width: 72px;
        height: 72px;
        background: linear-gradient(135deg, #FF9500 0%, #E68600 100%);
        border-radius: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 16px rgba(230, 134, 0, 0.3);
        position: relative;
    }
    
    .patch-logo::before {
        content: '';
        position: absolute;
        width: 22px;
        height: 22px;
        border: 2.5px solid rgba(255,255,255,0.5);
        border-radius: 4px;
        transform: translate(-4px, -4px);
    }
    
    .patch-logo::after {
        content: '';
        position: absolute;
        width: 22px;
        height: 22px;
        border: 2.5px solid #FFFFFF;
        border-radius: 4px;
        transform: translate(4px, 4px);
    }
    
    .patch-title {
        font-size: 48px !important;
        font-weight: 600;
        color: #1A1A1A;
        margin: 0 !important;
        padding: 0 !important;
        letter-spacing: -0.5px;
        line-height: 0.9 !important;
    }

    .patch-subtitle {
        font-size: 20px !important;
        color: #666;
        margin: 0 !important;
        padding: 0 !important;
        margin-top: 4px !important;
        font-weight: 400;
    }
    
    /* Chat card */
    .chat-card {
        background: #FFFFFF;
        border-radius: 20px;
        box-shadow: 0 2px 40px rgba(0,0,0,0.06);
        padding: 32px;
        margin-bottom: 24px;
    }
    
    /* Message bubbles */
    .message-user {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 16px;
    }
    
    .message-assistant {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 16px;
    }
    
    .bubble-user {
        max-width: 85%;
        padding: 16px 20px;
        font-size: 15px;
        line-height: 1.6;
        background: #1A1A1A;
        color: #FFFFFF;
        border-radius: 20px 20px 4px 20px;
    }
    
    .bubble-assistant {
        max-width: 85%;
        padding: 16px 20px;
        font-size: 15px;
        line-height: 1.6;
        background: #F5F5F5;
        color: #1A1A1A;
        border-radius: 20px 20px 20px 4px;
    }
    
    /* Input container - makes button appear inside */
    [data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
        position: relative;
        overflow: visible !important;
    }
    
    /* Form columns container */
    [data-testid="stForm"] > div:first-child {
        overflow: visible !important;
    }
    
    /* Input styling - clean edges, no shadow */
    .stTextInput > div > div > input {
        border: 2px solid #EBEBEB !important;
        border-radius: 28px !important;
        padding: 16px 60px 16px 24px !important;
        font-size: 15px !important;
        font-family: 'Inter', sans-serif !important;
        background: #FFFFFF !important;
        height: 56px !important;
        box-sizing: border-box !important;
        box-shadow: none !important;
    }
    
    .stTextInput > div {
        overflow: visible !important;
        box-shadow: none !important;
        border: none !important;
        background: transparent !important;
    }
    
    .stTextInput > div > div {
        box-shadow: none !important;
        border: none !important;
        background: transparent !important;
    }
    
    .stTextInput {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }
    
    /* Kill all baseweb input styling */
    [data-baseweb="input"] {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }
    
    [data-baseweb="base-input"] {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }
    
    div[data-baseweb="input"] > div {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #EBEBEB !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    /* Remove any focus ring */
    *:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    
    /* Position the form columns to overlap */
    [data-testid="stForm"] [data-testid="column"]:last-child {
        position: absolute !important;
        right: 8px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        width: auto !important;
        flex: none !important;
    }
    
    /* Send button - circular with arrow */
    .stFormSubmitButton > button {
        background: #1A1A1A !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 50% !important;
        width: 40px !important;
        height: 40px !important;
        padding: 0 !important;
        font-size: 18px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        min-width: 40px !important;
    }
    
    .stFormSubmitButton > button:hover {
        background: #333 !important;
    }
    
    .stFormSubmitButton > button p {
        margin: 0 !important;
        line-height: 1 !important;
    }
    
    /* Regular buttons - black, smaller */
    .stButton > button {
        background: #1A1A1A !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 6px 12px !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stButton > button:hover {
        background: #333 !important;
    }
    
    /* Footer */
    .patch-footer {
        text-align: center;
        font-size: 13px;
        color: #999;
        margin: 24px 0 16px 0;
    }
    
    /* Admin panel */
    .admin-panel {
        background: #FFFFFF;
        border-radius: 16px;
        box-shadow: 0 2px 40px rgba(0,0,0,0.06);
        padding: 24px;
        margin-top: 16px;
    }
    
    .admin-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
    }
    
    .admin-title {
        font-size: 14px;
        font-weight: 600;
        color: #1A1A1A;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 0;
    }
    
    .admin-badge {
        font-size: 12px;
        color: #999;
        background: #F5F5F5;
        padding: 4px 10px;
        border-radius: 100px;
    }
</style>
""", unsafe_allow_html=True)


# ---------- RAG SETUP ----------
@st.cache_resource
def load_qa_chain():
    if not CHROMA_DIR.exists():
        return None
    
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )

    llm = Ollama(model=MODEL_NAME)
    
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
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain


def rebuild_knowledge_base():
    loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    
    st.cache_resource.clear()
    return len(documents), len(chunks)


def get_doc_count():
    if DATA_DIR.exists():
        return len(list(DATA_DIR.glob("*.txt")))
    return 0


def build_enhanced_query(messages, current_question):
    """
    Build a query that includes recent conversation context.
    This helps the retriever find relevant documents for follow-up questions.
    """
    # Get the last 2-3 exchanges (excluding the current question we just added)
    # Each exchange = 1 user message + 1 assistant response
    recent_messages = messages[-(5):-1] if len(messages) > 1 else []  # Up to 4 previous messages (2 exchanges)

    if not recent_messages:
        return current_question

    # Build context string from recent conversation
    context_parts = []
    for msg in recent_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        context_parts.append(f"{role}: {msg['content']}")

    context_str = "\n".join(context_parts)

    # Create enhanced query with context
    enhanced_query = f"""Previous conversation:
{context_str}

Current question: {current_question}"""

    return enhanced_query


# ---------- MAIN APP ----------
def main():
    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi, I'm Patch.\nWhat can I fix for you today?"}
        ]
    
    if "show_admin" not in st.session_state:
        st.session_state.show_admin = False

    qa_chain = load_qa_chain()
    
   # Header with logo image
    logo_path = Path(__file__).parent / "logo.png"
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_base64 = base64.b64encode(f.read()).decode()
        
        st.markdown(f"""
        <div class="patch-header">
            <img src="data:image/png;base64,{logo_base64}" style="height: 80px;" alt="Patch logo">
            <div>
                <h1 class="patch-title">Patch</h1>
                <p class="patch-subtitle">Your IT support assistant</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Build messages HTML
    messages_html = ""
    for msg in st.session_state.messages:
        content = msg["content"].replace("\n", "<br>")
        if msg["role"] == "user":
            messages_html += f'<div class="message-user"><div class="bubble-user">{content}</div></div>'
        else:
            messages_html += f'<div class="message-assistant"><div class="bubble-assistant">{content}</div></div>'
    
    # Chat card with messages
    st.markdown(f"""
    <div class="chat-card">
        {messages_html}
    </div>
    """, unsafe_allow_html=True)
    
    # Input form with circular send button
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([10, 1])
        
        with col1:
            user_input = st.text_input(
                "Message",
                placeholder="Describe your issue...",
                label_visibility="collapsed"
            )
        
        with col2:
            send_clicked = st.form_submit_button("↑", use_container_width=True)
    
    # New chat button with spacing
    st.markdown('<div style="margin-top: 12px;"></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2.5, 1, 2.5])
    with col2:
        if st.button("+ New chat", key="new_chat", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": "Hi, I'm Patch.\nWhat can I fix for you today?"}
            ]
            st.rerun()
    
    # Process input
    if send_clicked and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        if qa_chain:
            with st.spinner("Thinking..."):
                # Build enhanced query with conversation context for better retrieval
                enhanced_query = build_enhanced_query(st.session_state.messages, user_input)
                result = qa_chain.invoke({"query": enhanced_query})
                response = result["result"]
                st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "I'm not connected to a knowledge base yet. Please set one up in the admin panel below."
            })
        
        st.rerun()
    
    # Footer
    st.markdown('<div class="patch-footer">Powered by local AI · Your data stays private</div>', unsafe_allow_html=True)
    
    # Admin toggle
    col1, col2, col3 = st.columns([5, 1, 5])
    with col2:
        if st.button("⚙️", key="admin_toggle", help="Admin settings"):
            st.session_state.show_admin = not st.session_state.show_admin
            st.rerun()
    
    # Admin panel
    if st.session_state.show_admin:
        doc_count = get_doc_count()
        
        st.markdown(f"""
        <div class="admin-panel">
            <div class="admin-header">
                <h3 class="admin-title">Admin</h3>
                <span class="admin-badge">{doc_count} documents loaded</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["txt"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = DATA_DIR / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"✓ Uploaded: {uploaded_file.name}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Rebuild Knowledge Base", use_container_width=True):
                with st.spinner("Building..."):
                    try:
                        doc_count, chunk_count = rebuild_knowledge_base()
                        st.success(f"✓ Built from {doc_count} docs ({chunk_count} chunks)")
                    except Exception as e:
                        st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
