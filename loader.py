import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredXMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_medical_documents(directory: str):
    """Load medical documents from TXT, PDF, and XML files."""
    all_docs = []

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        # Load text files
        if filename.lower().endswith(".txt"):
            loader = TextLoader(filepath)
            docs = loader.load()

        # Load PDF files
        elif filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            docs = loader.load()

        # Load XML files
        elif filename.lower().endswith(".xml"):
            loader = UnstructuredXMLLoader(filepath)
            docs = loader.load()

        # Skip other file types
        else:
            print(f"Skipping unsupported file type: {filename}")
            continue

        # Optionally print loaded file metadata
        for doc in docs:
            doc.metadata["source"] = filepath

        all_docs.extend(docs)

    # After loading, split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunked_docs = splitter.split_documents(all_docs)

    return chunked_docs
