"""
Docstring for chatbot.src.retriever
"""

"""this is a retriever file ok?"""

from langchain_pinecone import PineconeVectorStore
from gemini_ingest import get_vectordb
from gemini_ingest import get_embeddings
from langsmith import traceable
## "test-demo-gemini"
def get_vectorstore(index_name:str):
    """
    returns retriever object from vectorDB
    """

    index=get_vectordb(index_name)

    embeddings=get_embeddings()

    print("EMBEDDING DONE")

    vector_store=PineconeVectorStore(index=index,
                                     embedding=embeddings)
    
    print("Vector Store Initialised...")
    
    return vector_store


@traceable(name='RAG Retrieval Gemini Embedding')
def retrieve_context(query,top_k=4):
    """
    returns context from vectorDB
    """
    vector_store=get_vectorstore("test-demo-gemini")


    if vector_store is None:
        return "Error: Vector Store not initialised"

    try:
        retrieved_docs=vector_store.similarity_search_with_score(query,k=top_k)
        if not retrieved_docs:
            return "No relevant documents found",[]
        

        serialized = "\n\n".join(
            f"""
    -----------------------------------
        Score       : {score:.4f}
        Page number  : {doc.metadata.get("page_label", "N/A")}
        Total Pages : {int(doc.metadata.get("total_pages", -1))}
        Source      : {doc.metadata.get("source")}
        Content:
        {doc.page_content.strip()}
        """.strip()
            for i, (doc, score) in enumerate(retrieved_docs)
        )
        # s

        structured_docs = [
    {
        # "chunk_id": doc.id,
        "score": float(score),
        # "page": int(doc.metadata.get("page", -1)),
        "page_label": doc.metadata.get("page_label"),
        "total_pages": int(doc.metadata.get("total_pages", -1)),
        "source": doc.metadata.get("source"),
        "title": doc.metadata.get("title"),
        "content": doc.page_content
    }
    for doc, score in retrieved_docs
]

        return serialized,structured_docs

    except Exception as e:
        return f"Error: {e}",[]
    



if __name__ == "__main__":
    serialized,docs=retrieve_context("What is AI?")

    print(serialized)
    print(docs)