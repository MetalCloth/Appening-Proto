"""
Docstring for chatbot.src.graph
"""


"""
this is a graph file ok?
"""


from langchain_pinecone import PineconeVectorStore
from retriever import get_vectorstore,retrieve_context
from langsmith import traceable
from gemini_ingest import get_vectordb
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph import LangGraph
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from pydantic import BaseModel
load_dotenv()

import os
os.environ['GOOGLE_API_KEY']=os.getenv("GOOGLE_API_KEY")

chat_model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')



