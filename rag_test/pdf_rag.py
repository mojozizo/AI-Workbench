import os
import shutil
import hashlib
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Replace deprecated imports with the recommended ones
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
import chromadb
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

# Define paths
CHROMA_PATH = "chroma/pdf_collection"
DATA_PATH = "data"
COLLECTION_NAME = "pdf_documents"

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PDF RAG Application")
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--query", type=str, help="Query the RAG system.")
    args = parser.parse_args()
    
    if args.reset:
        print("✨ Clearing PDF Database")
        clear_database()

    # Create or update the data store
    if not args.query:
        # Process PDFs
        pdf_documents = load_pdf_documents()
        if pdf_documents:
            pdf_chunks = split_documents(pdf_documents)
            add_to_chroma(pdf_chunks)
            print("✅ PDF documents processed and added to the database")
        else:
            print("❌ No PDF documents found in the data directory")
    else:
        # Query the RAG system
        result = query_rag(args.query)
        print(f"\nQuery: {args.query}")
        print(f"\nResponse: {result}")

def load_pdf_documents():
    """Load all PDF documents from the data directory"""
    documents = []
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    
    for file_name in pdf_files:
        file_path = os.path.join(DATA_PATH, file_name)
        try:
            loader = PyPDFLoader(file_path)
            pdf_docs = loader.load()
            documents.extend(pdf_docs)
            print(f"Loaded PDF: {file_path} with {len(pdf_docs)} pages")
        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
    
    return documents

def split_documents(documents):
    """Split documents into chunks for better retrieval"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return text_splitter.split_documents(documents)

def generate_chunk_id(chunk_text, source, page):
    """Generate a unique ID for each document chunk"""
    unique_string = f"{source}-{page}-{chunk_text[:100]}"
    return hashlib.sha256(unique_string.encode()).hexdigest()

def add_to_chroma(chunks):
    """Add document chunks to ChromaDB"""
    # Initialize embedding function
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Create Chroma vector store
    db = Chroma(
        client=client,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME
    )

    # Add chunk IDs to metadata
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        chunk.metadata["id"] = generate_chunk_id(chunk.page_content, source, page)

    # Check for existing documents to avoid duplicates
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"]) if existing_items["ids"] else set()
    print(f"Number of existing documents in database: {len(existing_ids)}")

    # Only add documents that don't exist in the DB
    new_chunks = [chunk for chunk in chunks if chunk.metadata["id"] not in existing_ids]
    
    if new_chunks:
        print(f"Adding {len(new_chunks)} new document chunks to the database")
        # Add documents in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i + batch_size]
            batch_ids = [chunk.metadata["id"] for chunk in batch]
            db.add_documents(batch, ids=batch_ids)
    else:
        print("No new document chunks to add")

def clear_database():
    """Clear the database by removing the ChromaDB directory"""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("PDF database cleared successfully")

def query_rag(query_text):
    """Query the RAG system using the PDF collection"""
    # Initialize embedding function
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Check if collection exists
    collections = client.list_collections()
    # Update for ChromaDB v0.6.0+ compatibility
    collection_names = collections
    
    if COLLECTION_NAME not in collection_names:
        return "No PDF collection available. Please add documents first."
    
    # Load the collection
    db = Chroma(
        client=client,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME
    )
    
    # Create a retriever that fetches relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 5})
    
    # Initialize the LLM
    llm = OllamaLLM(model="stablelm2:latest")
    
    # Create prompt template for generating responses
    template = """
    Answer the question based only on the following context from PDF documents.
    If the information isn't in the context, say "I don't have that information in the PDF documents."
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    prompt = PromptTemplate.from_template(template)
    
    # Set up the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Execute the chain
    try:
        return rag_chain.invoke(query_text)
    except Exception as e:
        return f"Error querying PDF collection: {e}"

if __name__ == "__main__":
    main()