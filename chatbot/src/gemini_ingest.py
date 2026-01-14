"""Loads PDF
Extracts text
Chunks text (RecursiveCharacterTextSplitter)
Creates embeddings (Sentence Transformer all-MiniLM-L6-v2)
Stores in Pinecone
Main function: ingest_pdf()"""

# from PyPDF2 import PdfReader
from pypdf import PdfReader
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_pinecone import Pinecone,PineconeVectorStore
from pinecone import Pinecone,ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings

import os
from dotenv import load_dotenv
load_dotenv()

def get_embeddings():
    """Initializes embeddings for sentence-transformer"""
    # return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    api=os.getenv("GOOGLE_API_KEY")
    if not api:
        raise ValueError("API NOT FOUND")

    return GoogleGenerativeAIEmbeddings(model='gemini-embedding-001',api_key=api)

    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')




def load_document(path:str):
    """Loads document from path based on extension"""
    if not os.path.exists(path):
        raise ValueError(f"File {path} does not exist")

    ext=os.path.splitext(path)[1]

    if ext=='.pdf':
        try:
            print("Attempting to load pdf",{path})
            loader=PyPDFLoader(path)
        
        except Exception as e:
            print("Failed to load pdf",e)

    else:
        print("extension is",ext)
        
    return loader.load()


def get_vectordb(index_name:str):
    """Initializes vectorDB"""
    api_key=os.getenv('PINECONE_API_KEY')
    if not api_key:
        raise ValueError("API NOT FOUND")
    
    pc=Pinecone(api_key=api_key)

    if not pc.has_index(index_name):
        print("Index not found, creating index")
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        time.sleep(2)
    
    index=pc.Index(index_name)
    return index


def ingest_document(path:str):
    """Ingests document"""

    try:
        docs=load_document(path)
        print(f"Loaded {len(docs)} page documents")
        print("Now proceeding to splitting document")
        splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        splits=splitter.split_documents(docs)
        print(f"Created {len(splits)} chunks")

        print("Initializing embeddings")

        embeddings=get_embeddings()

        print("Getting index name")

        index=get_vectordb("test-demo-gemini")

        print("Getting vector store")

        vector_store=PineconeVectorStore(
            index=index,
            embedding=embeddings
        )
        print("Adding documents to vector store in batches...")

        batch_size = 20  # Keep small for free tier
        for i in range(0, len(splits), batch_size):
            batch = splits[i : i + batch_size]
            print(f"Processing batch {i//batch_size + 1} of {(len(splits)//batch_size)+1}...")
            
            try:
                vector_store.add_documents(batch)
                print("  - Batch Success. Sleeping 2s...")
                time.sleep(2) # Crucial sleep to reset rate limit
            except Exception as e:
                print(f"  - Error on batch: {e}")
                print("  - Waiting 30s for cooldown...")
                time.sleep(30)
                vector_store.add_documents(batch)
        # ----------------------------------

        print("Ingestion complete!")
        print("Adding documents to vector store")

        # vector_store.add_documents(splits)

    except Exception as e:
        print("Failed to load document",e)

    return docs


if __name__=="__main__":
    ingest_document(r"C:\Users\rawat\lemon\chatbot\data\Ebook-Agentic-AI.pdf")


