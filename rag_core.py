from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama


def build_rag(documents):
    """Build a RAG chain from documents."""
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create vector store
    faiss_db = FAISS.from_documents(documents, embeddings)
    retriever = faiss_db.as_retriever()

    # Initialize LLM
    llm = Ollama(
        model="llama3.1:8b",
        temperature=0
    )

    # Create RAG chain
    def rag_query(query):
        # Retrieve relevant documents
        docs = retriever.invoke(query)
        
        # Format context from documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create prompt
        prompt_text = f"""Use the following context to answer the question:

{context}

Question: {query}
Answer:"""
        
        # Get response from LLM
        response = llm.invoke(prompt_text)
        
        # Return in format expected by Streamlit UI
        return {
            "result": response,
            "source_documents": docs
        }

    return rag_query
