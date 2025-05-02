# app/rag_engine.py

import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from sentence_transformers import SentenceTransformer


class RAGEngine:
    def __init__(self):
        self.generator = pipeline("text2text-generation", model="t5-small", tokenizer="t5-small")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.nlp = spacy.load("en_core_web_sm")
        self.chunk_size = 150
        self.chunk_overlap = 30
        self.document_chunks = []
        self.chunk_embeddings = None

    def chunk_text(self, text):
        """
        Splits the input text into overlapping chunks using spaCy tokens.
        """
        doc = self.nlp(text)
        tokens = [token.text for token in doc]
        chunks = []

        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(tokens[i:i + self.chunk_size])
            chunks.append(chunk)

        self.document_chunks = chunks
        return chunks

    def build_index(self, chunks):
        """
        Builds an embedding index for all text chunks.
        """
        self.chunk_embeddings = self.embedding_model.encode(chunks, convert_to_tensor=True)
        self.document_chunks = chunks

    def retrieve_relevant_chunks(self, query, top_k=3):
        """
        Returns the top-k relevant chunks using cosine similarity.
        """
        if self.chunk_embeddings is None:
            raise ValueError("Index has not been built. Call build_index() first.")

        query_embedding = self.embedding_model.encode([query], convert_to_tensor=True)
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_chunks = [self.document_chunks[i] for i in top_indices]
        return top_chunks

    def generate_answer(self, query, context):
        """
        Uses the generator model to produce an answer from query and context.
        """
        prompt = f"question: {query} context: {context}"
        result = self.generator(prompt, max_length=256, do_sample=False)
        return result[0]["generated_text"]

    def query(self, query):
        """
        Returns the answer and supporting chunks.
        """
        top_chunks = self.retrieve_relevant_chunks(query)
        context = " ".join(top_chunks)
        answer = self.generate_answer(query, context)
        return answer, top_chunks
