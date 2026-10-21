"""Demo 2: RAG (Retrieval-Augmented Generation) with Ollama + ChromaDB.

The model answers questions using only the contents of one document:

  1. CHUNK    - split the document into small overlapping pieces 
  2. EMBED    - turn each piece into a vector with an Ollama embedding model 
  3. STORE    - save the vectors in a vector database (ChromaDB)
  4. RETRIEVE - embed the question and find the most similar pieces 
  5. GENERATE - give those pieces to the chat model and ask it to answer

Run:  python rag.py                (uses document.txt)
      python rag.py my_notes.txt   (use your own text file)
"""

import sys

import chromadb
import ollama

CHAT_MODEL = "gemma3:1b"
EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE = 500     # characters per chunk
CHUNK_OVERLAP = 100  # characters shared between neighbouring chunks
TOP_K = 3            # how many chunks to retrieve per question


def chunk_text(text):
    """Split text into overlapping chunks so a fact cut at a boundary still appears whole."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed(texts):
    """Turn a list of strings into a list of vectors."""
    return ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]


# --- 1-3. Chunk, embed and store the document --------------------------------

doc_path = sys.argv[1] if len(sys.argv) > 1 else "document.txt"
with open(doc_path, encoding="utf-8") as f:
    chunks = chunk_text(f.read())

# In-memory database, rebuilt on every run. To keep it on disk instead, use
# chromadb.PersistentClient(path="chroma_db").
db = chromadb.Client()
collection = db.create_collection("docs", configuration={"hnsw": {"space": "cosine"}})
collection.add(
    ids=[str(i) for i in range(len(chunks))],
    documents=chunks,
    embeddings=embed(chunks),
)
print(f"Indexed {doc_path}: {len(chunks)} chunks stored in ChromaDB.")
print("Ask a question about the document. Type 'exit' to quit.\n")

# --- 4-5. Retrieve relevant chunks and generate an answer --------------------

while True:
    question = input("You: ").strip()
    if question.lower() in ("exit", "quit"):
        break
    if not question:
        continue

    results = collection.query(query_embeddings=embed([question]), n_results=TOP_K)
    retrieved = results["documents"][0]

    print("\n[Retrieved chunks]")
    for chunk_id, distance in zip(results["ids"][0], results["distances"][0]):
        print(f"  chunk {chunk_id} (distance {distance:.2f})")

    context = "\n\n---\n\n".join(retrieved)
    prompt = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, tell the user based on your knowledge.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    print("\nBot: ", end="", flush=True)
    for part in ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ):
        print(part["message"]["content"], end="", flush=True)
    print("\n")
