# **RAG (Retrieval-Augmented Generation) Process Explained**  
*Combining your understanding with clarifications and best practices.*

---

### **1. Document Loading**  
- **Goal**: Load unstructured/semi-structured documents (PDFs, CSVs, text files) into usable text format.  
- **Tools**: LangChain document loaders (e.g., `PyPDFLoader`, `CSVLoader`, `UnstructuredFileLoader`).  
- **Process**:  
  - **Example**: A PDF is split into pages, with metadata (page number, source file name).  
  - **Output**: A list of `Document` objects containing text and metadata.  

---

### **2. Text Splitting (Chunking)**  
- **Why**: Embedding models have token limits (e.g., 512–8192 tokens), and smaller chunks improve retrieval accuracy.  
- **How**:  
  - **Chunk Size**: Typically **500–1000 tokens** (or characters, depending on the model).  
  - **Overlap**: Add **10–20% overlap** between chunks to preserve context (e.g., `RecursiveCharacterTextSplitter` in LangChain).  
- **Example Code**:  
  ```python
  from langchain_text_splitters import RecursiveCharacterTextSplitter

  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,  
      chunk_overlap=200,  
      separators=["\n\n", "\n", " "]  # Split by paragraphs, then sentences
  )
  chunks = text_splitter.split_documents(documents)  # Split into smaller `Document` objects
  ```

---

### **3. Chunk Identification & Metadata**  
- **Chunk ID**: A unique identifier for each chunk.  
  - **How**: Generate a hash of the chunk content + metadata (e.g., SHA-256).  
    ```python
    import hashlib

    def generate_chunk_id(chunk_text, source, page):
        unique_string = f"{source}-{page}-{chunk_text}"
        return hashlib.sha256(unique_string.encode()).hexdigest()
    ```  
- **Metadata**: Attach additional context to each chunk:  
  - `source`: Original document name.  
  - `page_number`: Page in the source document (for PDFs).  
  - `timestamp`: When the chunk was created.  

---

### **4. Generating Embeddings**  
- **Critical Step**: Convert text chunks into numerical vectors (embeddings) using an embedding model.  
- **Models**:  
  - OpenAI: `text-embedding-3-small`  
  - Open-source: `all-MiniLM-L6-v2` (Hugging Face)  
- **Process**:  
  ```python
  from langchain_openai import OpenAIEmbeddings

  embeddings_model = OpenAIEmbeddings()
  chunk_texts = [chunk.page_content for chunk in chunks]
  embeddings = embeddings_model.embed_documents(chunk_texts)
  ```

---

### **5. Storing in ChromaDB (Vector Store)**  
- **Goal**: Store chunks, embeddings, and metadata for fast retrieval.  
- **Steps**:  
  1. **Initialize ChromaDB**:  
     ```python
     from langchain_community.vectorstores import Chroma

     # Create a persistent database
     vector_store = Chroma.from_texts(
         texts=chunk_texts,
         embedding=embeddings_model,
         metadatas=[chunk.metadata for chunk in chunks],  # Includes source, page, etc.
         ids=[generate_chunk_id(...) for chunk in chunks],  # Unique IDs
         persist_directory="./chroma_db"  # Save to disk
     )
     ```  
  2. **Avoid Duplicates**:  
     - Before adding a chunk, check if its ID already exists in the database.  
     - Use ChromaDB’s `get` method:  
       ```python
       existing_ids = vector_store.get(ids=[chunk_id])["ids"]
       if chunk_id not in existing_ids:
           vector_store.add_texts(...)
       ```

---

### **6. Querying the RAG System**  
- **At Runtime**:  
  1. **Embed the User Query**:  
     ```python
     query = "What is the capital of France?"
     query_embedding = embeddings_model.embed_query(query)
     ```  
  2. **Retrieve Relevant Chunks**:  
     ```python
     results = vector_store.similarity_search(query, k=3)  # Top 3 chunks
     ```  
  3. **Generate an Answer**: Pass the chunks + query to an LLM (e.g., GPT-4):  
     ```python
     from langchain_core.prompts import ChatPromptTemplate
     from langchain_openai import ChatOpenAI

     llm = ChatOpenAI(model="gpt-4")
     prompt = ChatPromptTemplate.from_template(
         "Answer using only this context: {context}\nQuestion: {question}"
     )
     chain = prompt | llm
     response = chain.invoke({"context": results, "question": query})
     ```
