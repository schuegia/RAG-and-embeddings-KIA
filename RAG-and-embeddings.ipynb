import os
import glob
import pickle
import numpy as np
import tqdm
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
import faiss
import umap.umap_ as umap
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from groq import Groq

# Einlesen und Verarbeiten der PDF-Dateien
pdf_paths = glob.glob("data/*.pdf")
full_text = ""
for path in tqdm.tqdm(pdf_paths, desc="Reading PDFs"):
    with open(path, "rb") as file:
        reader = PdfReader(file)
        full_text += " ".join(p.extract_text() for p in reader.pages if p.extract_text())

# Aufteilung in Textabschnitte
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_text(full_text)

# Zusätzliche Token-basierte Aufteilung
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=128, model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
split_chunks = []
for ch in chunks:
    split_chunks += token_splitter.split_text(ch)

# Embedding-Modell vorbereiten
embedding_model_name = "Sahajtomar/German-semantic"
embedding_model = SentenceTransformer(embedding_model_name)
chunk_vectors = embedding_model.encode(split_chunks, convert_to_numpy=True)

# Aufbau des FAISS-Index
dim = chunk_vectors.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(chunk_vectors)

# Speichern des Index und der Textzuordnung
os.makedirs("faiss", exist_ok=True)
faiss.write_index(index, "faiss/faiss_index.index")
with open("faiss/chunks_mapping.pkl", "wb") as f:
    pickle.dump(split_chunks, f)

# UMAP-Reduktion für Visualisierung
umap_model = umap.UMAP(random_state=0).fit(chunk_vectors)

def project(embeddings, model):
    result = np.empty((len(embeddings), 2))
    for i, emb in enumerate(tqdm.tqdm(embeddings, desc="Projecting")):
        result[i] = model.transform([emb])
    return result

proj_all = project(chunk_vectors, umap_model)

# Semantische Suche
def retrieve(query, k=5):
    query_vec = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k)
    texts = [split_chunks[i] for i in indices[0]]
    vectors = np.array([chunk_vectors[i] for i in indices[0]])
    return texts, vectors, distances[0]

# Visualisierung der Ergebnisse
def visualize(query, retrieved_texts, retrieved_vectors):
    proj_retrieved = project(retrieved_vectors, umap_model)
    proj_query = project(embedding_model.encode([query], convert_to_numpy=True), umap_model)

    plt.figure()
    plt.scatter(proj_all[:, 0], proj_all[:, 1], s=10, color='gray', label='Datenbasis')
    plt.scatter(proj_retrieved[:, 0], proj_retrieved[:, 1], s=100, edgecolors='green', facecolors='none', label='Treffer')
    plt.scatter(proj_query[:, 0], proj_query[:, 1], s=150, color='red', marker='X', label='Anfrage')

    def shorten(text, max_len=15):
        return (text[:max_len] + '...') if len(text) > max_len else text

    for i, txt in enumerate(retrieved_texts):
        plt.annotate(shorten(txt), (proj_retrieved[i, 0], proj_retrieved[i, 1]), fontsize=8)

    plt.annotate(shorten(query), (proj_query[0, 0], proj_query[0, 1]), fontsize=8)
    plt.title("Embedding-Projektion")
    plt.legend()
    plt.gca().set_aspect('equal', 'datalim')
    plt.show()

# Prompt-Erstellung für LLM
def build_prompt(question: str, context_blocks: list[str]) -> str:
    context = "\n\n".join(context_blocks)
    return f"""Beantworte folgende Frage basierend auf dem gegebenen Kontext.

Kontext:
{context}

Frage:
{question}

Antwort:"""

# Anfrage an das Groq-Modell
def query_llm(prompt: str) -> str:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content

# Komplette RAG-Pipeline
def run_pipeline(question: str, top_k: int = 5):
    print(f"\nFrage: {question}")
    retrieved, vectors, _ = retrieve(question, top_k)
    prompt = build_prompt(question, retrieved)
    print("\nSende Anfrage an LLM...")
    answer = query_llm(prompt)
    print("\nAntwort:\n")
    print(answer)
    visualize(question, retrieved, vectors)

# Beispielanfrage
if __name__ == "__main__":
    run_pipeline("Wie müssen KI-generierte Inhalte in einer Arbeit gekennzeichnet werden?")
