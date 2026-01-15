import fastapi
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import os
import sys
# from src.graph import run_graph, ChatRequest, ChatResponse


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from graph import run_graph,ChatRequest,ChatResponse

app = FastAPI(
    title="Agentic AI RAG Chatbot",
    description="RAG-based chatbot answering questions from Agentic AI ebook",
    version="1.0.0"
)


@app.get("/")
def root():
    """Root endpoint"""
    return{
        "status":"ok",
        "endpoints":{
            "POST /chat":"Ask a question",
            "GET /health": "Health check",
            "POST /sample_query": "Sample question pipeline"
        }
    }


@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "agentic-ai-chatbot"
    }


@app.post("/chat",response_model=ChatResponse)
async def chat(request:ChatRequest):
    """
    Main chat endpoint
    
    Request body:
    {
        "question": "What is Agentic AI?"
    }
    
    Response:
    {
        "answer": "...",
        "context": [...],
        "confidence": 0.85
    }
    """

    try:
        if not request.question or request.question.strip()=="":
            raise HTTPException(status_code=400,detail="Question can't be empty")
        
        response=run_graph(request.question)

        return {
            "answer":response.answer,
            "confidence":response.confidence,
            "context":response.context
        }

    except Exception as e:
        raise HTTPException(status_code=500,detail=f"Internal Error, {e}")
    


@app.get("/sample_query")
def sample_query():
    """Returns sample queries for testing"""

    return {
        "queries": [
            "What is Agentic AI?",
            "How does Agentic AI differ from traditional AI?",
            "What are multi-agent systems?",
            "What are the healthcare use cases for Agentic AI?",
            "What are the challenges of orchestrating multi-agent systems?",
            "How do you assess organizational readiness for Agentic AI?"
        ]
    }
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# @app.post("/ingest_docs")
# def ingest_docs(file_path:str):
#     print("Ingesting docs...")
#     try:

#         ingest_document(file_path)
    
#     except Exception as e:
#         print(e)
#     return{"ingested":True}



# @app.post('/query')
# def query(query:str)->dict:
#     return run_graph(query)
