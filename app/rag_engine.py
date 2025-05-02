import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

class RAGEngine:
    def __init__(self, embedding_model="all-MiniLM-L6-v2", qa_model="distilbert-base-cased-distilled-squad"):
        self.embedder = SentenceTransformer(embedding_model)
        self.qa_model = pipeline("question-answering", model=qa_model)
        self.index = None
        self.documents = []

    def chunk_text(self, text):
        raw_chunks = re.split(r'\n(?=\d+\s|\d+\.\d+|[A-Z].+:)', text)
        return [chunk.strip() for chunk in raw_chunks if len(chunk.strip()) > 200]

    def build_index(self, documents):
        self.documents = documents
        embeddings = self.embedder.encode(documents, show_progress_bar=True)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings, dtype=np.float32))

    def query(self, user_input, top_k=5):
        query_embedding = self.embedder.encode([user_input])
        D, I = self.index.search(np.array(query_embedding, dtype=np.float32), k=top_k)
        top_chunks = [self.documents[i] for i in I[0]]
        context = "\n\n".join(top_chunks)
        answer = self.qa_model(question=user_input, context=context)
        return answer['answer'], top_chunks