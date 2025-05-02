# app/rag_engine.py

import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import torch


class RAGEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device set to use {self.device}")

        # Embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Generator model setup
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small").to(self.device)

        # Chunking settings
        self.nlp = spacy.load("en_core_web_sm")
        self.chunk_size = 150
        self.chunk_overlap = 30
        self.document_chunks = []
        self.chunk_embeddings = None
        self.max_input_length = 512
        self.max_output_length = 200

    def chunk_text(self, text):
        """
        Splits input text into overlapping chunks using spaCy tokens.
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
        Builds an embedding index for the text chunks.
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
        return [self.document_chunks[i] for i in top_indices]

    def generate_answer(self, query, context):
        """
        Generates an answer using the T5 model from query and context.
        """
        prompt = f"question: {query} context: {context}"

        # Tokenize with truncation to avoid exceeding max length
        input_ids = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
            padding="max_length"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=self.max_output_length,
                num_beams=4,
                early_stopping=True
            )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()

    def query(self, query):
        """
        Takes user query, retrieves relevant chunks, generates answer.
        """
        top_chunks = self.retrieve_relevant_chunks(query)
        context = " ".join(top_chunks)
        answer = self.generate_answer(query, context)
        return answer, top_chunks
