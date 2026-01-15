"""
Docstring for chatbot.src.graph
"""


"""
this is a graph file ok?
"""


from langchain_pinecone import PineconeVectorStore
from prompts import RAG_PROMPT
from retriever import get_vectorstore,retrieve_context
from langsmith import traceable
from langchain_core.prompts import PromptTemplate
from gemini_ingest import get_vectordb
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,END,START
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
load_dotenv()

import os
os.environ['GOOGLE_API_KEY']=os.getenv("GOOGLE_API_KEY")
os.environ['OPENROUTER_API_KEY']=os.getenv('OPENROUTER_API_KEY')

class ChatRequest(BaseModel):
    question: str

class ContexChunk(BaseModel):
    content:str
    score:float
    page:Optional[int]=None


class ChatResponse(BaseModel):
    answer:str
    context:List[ContexChunk]
    confidence:float


class GraphState(TypedDict):
    query:str
    context:List[dict]
    answer:str
    serialized_content:str
    confidence:str


# chat_model=ChatGoogleGenerativeAI(model='gemini-2.0-flash')


chat_model=ChatOpenAI(
        api_key=os.getenv('OPENROUTER_API_KEY'),
        base_url="https://openrouter.ai/api/v1",
        model='google/gemini-2.5-flash',  
        max_tokens=4000,
        temperature=0.1,
)




@traceable(name="Retrieve Context - Pinecone")
def retrieve_node(state:GraphState)->GraphState:
    """Retrieve relevant context from Pinecone"""
    print("Retrieveing context from Pinecone...")

    serialized,structured=retrieve_context(state['query'],top_k=8)

    state['context']=structured
    state['serialized_content']=serialized

    print("Retrieved context from Pinecone...")

    return state


@traceable(name="Generate Answer - Gemini")
def generate_node(state:GraphState)->GraphState:
    """answering Node of LLM"""
    print("Generating answer...")

    prompt=RAG_PROMPT.format(
        context=state['serialized_content'],
        question=state['query']
    )

    response=chat_model.invoke(prompt)

    state['answer']=response.content

    print("Answer generated")

    return state

@traceable(name="Calculate Confidence Score")
def calculate_confidence_score(state:GraphState)->GraphState:
    """Calculating confidence score"""
    print("Calculating confidence score...")

    if not state['context']:
        state['confidence']=0.0
        return state
    
    top_scores=[doc['score'] for doc in state['context'][:3]]
    avg_score=sum(top_scores)/len(top_scores)
    
    state['confidence']=round(avg_score, 2)
    
    print(f" Confidence: {state['confidence']}")
    return state



@traceable(name="Create RAG Graph")
def create_rag_graph():
    """Creates and compiles RAG graph"""

    graph=StateGraph(GraphState)

    graph.add_node("retrieve",retrieve_node)
    graph.add_node("generate",generate_node)
    graph.add_node("confidence",calculate_confidence_score)

    graph.add_edge(START,"retrieve")
    graph.add_edge("retrieve","generate")
    graph.add_edge("generate","confidence")
    graph.add_edge("confidence",END)

    return graph.compile()



@traceable(name='RAG Graph Gemini Embedding')
def run_graph(query:str)->ChatResponse:
    """Run the RAG pipeline"""
    graph=create_rag_graph()

    initial_state=GraphState(
        query=query,
        context=[],
        serialized_content="",
        answer="",
        confidence=0.0
    )

    result=graph.invoke(initial_state)

    context_chunks = [
        ContexChunk(
            content=doc['content'] + "\n", 
            score=doc['score'],
            page=doc['page_label']
        )
        for doc in result['context']
    ]
    
    return ChatResponse(
        answer=result['answer'],
        context=context_chunks,
        confidence=result['confidence']
    )
import time


if __name__=="__main__":
    adversarial_queries = [
    # Off-topic
    "What is the weather today?",
    
    # Not in PDF
    "What is the pricing for Agentic AI implementation?",
    
    # Complex reasoning
    "If a retail company is at Level 2 with moderate data quality, what exact steps from the decision tree should they follow?",
    
    # Nuance
    "What are the disadvantages of Agentic AI?",
    
    # Table extraction
    "List ALL the types of agents with examples in a structured format",
    
    # Contradiction
    "Is Agentic AI just another name for LLMs?",
    
    # Ambiguous
    "How does it work?",
    
    # Multi-hop
    "Compare single-agent vs multi-agent systems, explain when to use each, and provide industry examples"
]
    hard_mode_queries = [
        # Test A: Deep Technical Retrieval (Page 21)
        "Explain the BDI model and how it relates to agent behavior.",
        
        # Test B: Entity Disambiguation (Page 60)
        "What is the specific role of Emergence AI compared to Konverge AI in this book?",
        
        # Test C: Precise Metrics Retrieval (Page 58)
        "What quantified impact did the Retail Copilot have on customer engagement and sales?"
    ]

    x=[
        "1. How does Agentic AI fundamentally differ from Traditional AI, RPA, and standard LLMs?",
        "2. Explain the BDI Model and how it drives an agent's behavior within the core pillars of Perception to Execution.",
        "3. In a Multi-Agent System (MAS), how does the Supply Chain in Crisis scenario demonstrate the superiority of MAS over single-agent systems?",
        "4. What are the specific challenges of orchestrating complex agentic systems, and how does the Orchestrator mitigate conflict and data security risks?",
        "5. According to the Organizational Maturity & Readiness Framework, what critical checkpoints must a company pass in the Decision Tree before implementing Agentic AI?"
    ]

    for q in x:
        print(f"\n{'='*70}")
        print(f"🎯 ADVERSARIAL QUERY: {q}")
        print('='*70)
        
        response = run_graph(q)
        
        print(f"\n📝 ANSWER:\n{response.answer}")
        print(f"\n🎲 CONFIDENCE: {response.confidence}")
        print(f"\n📃 CONTEXT:\n{response.context}")
        
        # Check for hallucination
        time.sleep(7)