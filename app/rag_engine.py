import fitz  # PyMuPDF
import os
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import networkx as nx


# Load PDF and extract text
def load_pdf_chunks(pdf_path, chunk_size=300, overlap=50):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    text = text.replace("\n", " ")
    
    # Split text into overlapping chunks
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# Embed chunks using MiniLM
def embed_chunks(chunks, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    return embeddings, embedder


# Build FAISS index
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


# Generate answer using FLAN-T5
def generate_answer(context_chunks, query, model_name="google/flan-t5-base"):
    prompt = "Answer the question based on the following context:\n"
    for chunk in context_chunks:
        prompt += f"- {chunk.strip()}\n"
    prompt += f"\nQuestion: {query}"
    
    generator = pipeline("text2text-generation", model=model_name, device=0 if torch.cuda.is_available() else -1)
    response = generator(prompt, max_length=256, do_sample=False)[0]['generated_text']
    return response.strip()


# RAG Pipeline
class RAGEngine:
    def __init__(self, pdf_path):
        print("[INFO] Loading and chunking PDF...")
        self.doc_chunks = load_pdf_chunks(pdf_path)
        print("[INFO] Embedding chunks...")
        self.embeddings, self.embedder = embed_chunks(self.doc_chunks)
        print("[INFO] Building FAISS index...")
        self.index = build_faiss_index(np.array(self.embeddings))

        # Initialize graph and chunk-entity mapping
        self.kg = nx.DiGraph()
        self.chunk_to_entity = {}

        # Simple dummy knowledge graph creation from chunks (you can customize this)
        for idx, chunk in enumerate(self.doc_chunks):
            entity = f"Entity{idx}"
            self.kg.add_node(entity)
            self.chunk_to_entity[idx] = entity

        # Initialize QA pipeline
        self.qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", device=0 if torch.cuda.is_available() else -1)

    def retrieve_chunks(self, question, top_k=5):
        query_embedding = self.embedder.encode([question], convert_to_numpy=True)
        D, I = self.index.search(query_embedding, k=top_k)

        relevant_entities = [n for n in self.kg.nodes if n.lower() in question.lower()]
        filtered_chunks = []

        for idx in I[0]:
            entity = self.chunk_to_entity.get(idx, None)
            if not relevant_entities or (entity and entity in relevant_entities):
                filtered_chunks.append(self.doc_chunks[idx])
            if len(filtered_chunks) >= top_k:
                break

        if not filtered_chunks:
            filtered_chunks = [self.doc_chunks[i] for i in I[0]]
        return filtered_chunks

    def query(self, question: str):
        top_chunks = self.retrieve_chunks(question)
        context = " ".join(top_chunks)
        input_text = f"question: {question} context: {context}"
        output = self.qa_pipeline(input_text, max_length=256, do_sample=False)
        answer = output[0]["generated_text"]
        return answer


# Example usage for terminal testing
if __name__ == "__main__":
    pdf_file = "/workspaces/rag-kg/data/Manual example - AeroCraft ACE-900.pdf"
    rag = RAGEngine(pdf_file)
    while True:
        query = input("\nAsk your question (or type 'exit'): ")
        if query.lower() == 'exit':
            break
        answer = rag.query(query)
        print(f"\nAnswer:\n{answer}")
