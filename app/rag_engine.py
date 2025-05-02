import torch
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


class RAGEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")

        # Embedding model
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(self.device))

        # Generator model using Hugging Face pipeline with Falcon-7B-Instruct
        model_name = "tiiuae/falcon-7b-instruct"
        print(f"[INFO] Loading model: {model_name}")
        self.generator = pipeline("text-generation", 
                                  model=AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(self.device),
                                  tokenizer=AutoTokenizer.from_pretrained(model_name),
                                  device=0 if torch.cuda.is_available() else -1)

        self.chunk_token_limit = 100
        self.chunk_overlap = 1
        self.document_chunks = []
        self.chunk_embeddings = None

    def chunk_text(self, text):
        sentences = sent_tokenize(text)
        chunks, current_chunk = [], []

        for sentence in sentences:
            current_chunk.append(sentence)
            token_count = sum(len(s.split()) for s in current_chunk)
            if token_count > self.chunk_token_limit:
                chunks.append(" ".join(current_chunk))
                current_chunk = current_chunk[-self.chunk_overlap:]  # Retain overlap

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        self.document_chunks = chunks
        return chunks

    def build_index(self, chunks):
        print("[INFO] Encoding document chunks...")
        self.chunk_embeddings = self.embedding_model.encode(
            chunks, convert_to_tensor=True, normalize_embeddings=True
        )
        self.document_chunks = chunks

    def retrieve_relevant_chunks(self, query, top_k=3):
        if self.chunk_embeddings is None:
            raise ValueError("Index not built. Run build_index() after chunking.")

        formatted_query = f"query: {query}"
        query_embedding = self.embedding_model.encode(
            formatted_query, convert_to_tensor=True, normalize_embeddings=True
        )
        scores = util.cos_sim(query_embedding, self.chunk_embeddings)[0]
        top_indices = torch.topk(scores, k=top_k).indices
        return [self.document_chunks[i] for i in top_indices]

    def generate_answer(self, query, context):
        prompt = f"""You are a knowledgeable assistant. Answer concisely based only on the context below.

Context:
{context}

Question: {query}
Answer:"""

        outputs = self.generator(prompt, max_new_tokens=256, do_sample=False, num_beams=4)
        return outputs[0]['generated_text'].split("Answer:")[-1].strip()

    def query(self, query):
        top_chunks = self.retrieve_relevant_chunks(query)
        context = " ".join(top_chunks)
        answer = self.generate_answer(query, context)
        return answer, top_chunks

