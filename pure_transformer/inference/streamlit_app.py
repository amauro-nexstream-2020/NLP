"""
Pure Transformer Chat UI

A modern Streamlit chat interface for the Pure Transformer LLM.
Features a dark glassmorphism design with responsive controls.
"""

import streamlit as st
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, "/workspace/NLP")

from pure_transformer.inference import load_model_from_checkpoint, generate_text, get_latest_checkpoint

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Pure Transformer Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Custom CSS for Glassmorphism Dark Theme
# =============================================================================

st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables */
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-glass: rgba(255, 255, 255, 0.03);
        --border-glass: rgba(255, 255, 255, 0.08);
        --text-primary: #ffffff;
        --text-secondary: #a0a0b0;
        --accent-purple: #8b5cf6;
        --accent-blue: #3b82f6;
        --accent-gradient: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%);
    }
    
    /* Global styles */
    .stApp {
        background: var(--bg-primary);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1200px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border-glass);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-primary);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: var(--text-primary) !important;
        font-weight: 600;
    }
    
    /* Chat message containers */
    .chat-message {
        padding: 1.5rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-glass);
    }
    
    .chat-message.user {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(59, 130, 246, 0.15) 100%);
        border-left: 3px solid var(--accent-purple);
    }
    
    .chat-message.assistant {
        background: var(--bg-glass);
        border-left: 3px solid var(--accent-blue);
    }
    
    .chat-message .message-content {
        color: var(--text-primary);
        line-height: 1.6;
    }
    
    .chat-message .message-role {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-secondary);
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    /* Input styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        padding: 1rem !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--accent-purple) !important;
        box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: var(--accent-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.3) !important;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: var(--accent-gradient) !important;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .status-badge.online {
        background: rgba(34, 197, 94, 0.2);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    .status-badge.loading {
        background: rgba(234, 179, 8, 0.2);
        color: #eab308;
        border: 1px solid rgba(234, 179, 8, 0.3);
    }
    
    /* Stats card */
    .stats-card {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .stats-card h4 {
        color: var(--text-secondary);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    
    .stats-card .value {
        color: var(--text-primary);
        font-size: 1.25rem;
        font-weight: 600;
    }
    
    /* Divider */
    hr {
        border-color: var(--border-glass) !important;
        margin: 1.5rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Model Loading (Cached)
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_model():
    """Load model and tokenizer (cached)."""
    checkpoint_path = get_latest_checkpoint("/checkpoints")
    if checkpoint_path is None:
        st.error("No checkpoint found in /checkpoints")
        return None, None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_from_checkpoint(
        checkpoint_path,
        model_size="xlarge",
        device=device
    )
    return model, tokenizer

# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Generation Settings")
    st.markdown("---")
    
    temperature = st.slider(
        "🌡️ Temperature",
        min_value=0.1,
        max_value=2.0,
        value=0.8,
        step=0.1,
        help="Higher = more creative, Lower = more focused"
    )
    
    top_k = st.slider(
        "🎯 Top-K",
        min_value=1,
        max_value=100,
        value=50,
        step=5,
        help="Number of top tokens to sample from"
    )
    
    max_tokens = st.slider(
        "📝 Max Tokens",
        min_value=32,
        max_value=512,
        value=256,
        step=32,
        help="Maximum tokens to generate"
    )
    
    st.markdown("---")
    st.markdown("## 📊 Model Info")
    
    # Model status
    model, tokenizer = load_model()
    
    if model is not None:
        st.markdown('<span class="status-badge online">● Online</span>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="stats-card">
            <h4>Model</h4>
            <div class="value">Pure Transformer 1.5B</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="stats-card">
            <h4>Device</h4>
            <div class="value">{"CUDA" if torch.cuda.is_available() else "CPU"}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge loading">● Loading...</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🔗 Quick Links")
    st.markdown("- [W&B Dashboard](https://wandb.ai)")
    st.markdown("- [GitHub Repo](https://github.com/amauro-nexstream-2020/NLP)")

# =============================================================================
# Main Content
# =============================================================================

st.markdown("""
<h1 style="background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%); 
           -webkit-background-clip: text; -webkit-text-fill-color: transparent;
           font-size: 2.5rem; margin-bottom: 0.5rem;">
    🤖 Pure Transformer Chat
</h1>
<p style="color: #a0a0b0; font-size: 1.1rem; margin-bottom: 2rem;">
    Chat with the Pure Transformer 1.5B language model trained on FineWeb
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    role_class = "user" if message["role"] == "user" else "assistant"
    role_label = "You" if message["role"] == "user" else "Pure Transformer"
    
    st.markdown(f"""
    <div class="chat-message {role_class}">
        <div class="message-role">{role_label}</div>
        <div class="message-content">{message["content"]}</div>
    </div>
    """, unsafe_allow_html=True)

# Chat input
prompt = st.chat_input("Enter your prompt...", key="chat_input")

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    st.markdown(f"""
    <div class="chat-message user">
        <div class="message-role">You</div>
        <div class="message-content">{prompt}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate response
    if model is not None:
        with st.spinner("✨ Generating..."):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            response = generate_text(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                device=device
            )
            
            # Remove prompt from response if present
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display assistant message
        st.markdown(f"""
        <div class="chat-message assistant">
            <div class="message-role">Pure Transformer</div>
            <div class="message-content">{response}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Model not loaded. Please check the checkpoint path.")

# Clear chat button
if st.session_state.messages:
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
