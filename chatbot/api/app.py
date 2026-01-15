import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from graph import run_graph

st.set_page_config(page_title="Agentic AI Chatbot", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Agentic AI Assistant")

# Sidebar
with st.sidebar:
    st.header("Quick Questions")
    
    questions = [
        "How does Agentic AI fundamentally differ from Traditional AI, RPA, and standard LLMs?",
        "Explain the BDI Model and how it drives an agent's behavior within the core pillars of Perception to Execution.",
        "In a Multi-Agent System (MAS), how does the Supply Chain in Crisis scenario demonstrate the superiority of MAS over single-agent systems?",
        "What are the specific challenges of orchestrating complex agentic systems, and how does the Orchestrator mitigate conflict and data security risks?",
        "According to the Organizational Maturity & Readiness Framework, what critical checkpoints must a company pass in the Decision Tree before implementing Agentic AI?"
    ]
    
    if "selected" not in st.session_state:
        st.session_state.selected = ""
    
    for i, q in enumerate(questions):
        if st.button(q, key=f"q_{i}"):
            st.session_state.selected = q

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant" and "meta" in msg:
            st.info(f"**Confidence Score: {msg['meta']['confidence']:.2f}**")
            
            with st.expander("📚 View Context"):
                context_json = msg["meta"]["context"]
                st.json(context_json)

# Input
user_input = st.chat_input("Ask anything about Agentic AI...")

if st.session_state.selected:
    user_input = st.session_state.selected
    st.session_state.selected = ""

if user_input:
    # Add user message
    import time
    msg_id = str(int(time.time() * 1000))
    st.session_state.messages.append({"role": "user", "content": user_input, "id": msg_id})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Call your graph
                response = run_graph(user_input)
                
                # Display answer
                st.markdown(response.answer)
                
                # Display confidence
                st.info(f"**Confidence Score: {response.confidence:.2f}**")
                
                # Display context in expander
                with st.expander("📚 View Context"):
                    context_list = [
                        chunk.dict() if hasattr(chunk, 'dict') else chunk 
                        for chunk in response.context
                    ]
                    st.json(context_list)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "id": msg_id,
                    "meta": {
                        "confidence": response.confidence,
                        "context": context_list
                    }
                })
                
            except Exception as e:
                st.error(f"Error: {str(e)}")