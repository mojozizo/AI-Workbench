# test.py - Converted from test.ipynb
# Covers both PDF RAG and CSV RAG workflows using ChromaDB

# ============================================================
# SECTION 1: PDF RAG
# ============================================================

from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
import chromadb

# --- Load PDF documents ---
def load_documents():
    document_loader = PyPDFDirectoryLoader(path="data")
    return document_loader.load()

documents = load_documents()

# --- Split documents ---
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

chunks = split_documents(documents)
print(chunks)

# --- Calculate chunk IDs ---
def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks

chunks_with_ids = calculate_chunk_ids(chunks)
print(chunks_with_ids)

# --- Persist chunks to ChromaDB (PDF collection) ---
persistent_client = chromadb.PersistentClient(path="chromadb")
collection = persistent_client.get_or_create_collection(name="my_documents")

existing_items = collection.get(include=[])
existing_ids = set(existing_items["ids"])

new_chunks = [chunk for chunk in chunks_with_ids
              if chunk.metadata["id"] not in existing_ids]

if new_chunks:
    pdf_documents = [chunk.page_content for chunk in new_chunks]
    metadatas = [chunk.metadata for chunk in new_chunks]
    ids = [chunk.metadata["id"] for chunk in new_chunks]

    collection.add(
        documents=pdf_documents,
        ids=ids,
        metadatas=metadatas
    )
    print(f"Added {len(pdf_documents)} new documents")
else:
    print("No new documents to add")

print(collection.peek())

results = collection.query(
    query_texts=["what happens in the start of monopoly ?"],
    n_results=2
)
print(results)


# ============================================================
# SECTION 2: CSV RAG
# ============================================================

from langchain_community.document_loaders.csv_loader import CSVLoader

# --- Load CSV ---
loader = CSVLoader(file_path="ICD.csv")
data = loader.load()
print(data)

# --- Split CSV documents ---
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

csv_chunks = split_documents(data)
print(csv_chunks)

# --- ChromaDB collection for ICD codes ---
persistent_client = chromadb.PersistentClient(path="chromadb")
collection = persistent_client.get_or_create_collection(name="icd_codes")

# --- Debug: Print metadata of first few chunks ---
for i, chunk in enumerate(csv_chunks[:3]):
    print(f"Chunk {i} metadata: {chunk.metadata}")

# --- Add unique IDs to CSV chunks ---
def calculate_csv_chunk_ids(chunks):
    row_counter = {}

    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source")
        row = chunk.metadata.get("row", i)

        row_key = f"{source}:row_{row}"
        if row_key in row_counter:
            row_counter[row_key] += 1
            chunk_id = f"{source}:row_{row}_{row_counter[row_key]}"
        else:
            row_counter[row_key] = 0
            chunk_id = f"{source}:row_{row}_0"

        chunk.metadata["id"] = chunk_id

    return chunks

csv_chunks_with_ids = calculate_csv_chunk_ids(csv_chunks)

for i, chunk in enumerate(csv_chunks_with_ids[:3]):
    print(f"Chunk {i} metadata after adding IDs: {chunk.metadata}")

ids = [chunk.metadata["id"] for chunk in csv_chunks_with_ids]
unique_ids = set(ids)
print(f"Total IDs: {len(ids)}, Unique IDs: {len(unique_ids)}")
if len(ids) == len(unique_ids):
    print("All IDs are unique!")
else:
    print(f"Found {len(ids) - len(unique_ids)} duplicate IDs")

# --- Batch-add CSV chunks to ChromaDB ---
if csv_chunks_with_ids:
    csv_documents = [chunk.page_content for chunk in csv_chunks_with_ids]
    metadatas = [chunk.metadata for chunk in csv_chunks_with_ids]
    ids = [chunk.metadata["id"] for chunk in csv_chunks_with_ids]

    batch_size = 100
    total_docs = len(csv_documents)

    for i in range(0, total_docs, batch_size):
        end_idx = min(i + batch_size, total_docs)
        print(f"Processing batch {i // batch_size + 1}/{(total_docs + batch_size - 1) // batch_size}: "
              f"documents {i} to {end_idx - 1}")
        collection.add(
            documents=csv_documents[i:end_idx],
            ids=ids[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )

    print(f"Added {total_docs} documents in batches")
else:
    print("No new documents to add")

print(collection.peek())

results = collection.query(
    query_texts=["cholera topica"],
    n_results=2
)
print(results)
