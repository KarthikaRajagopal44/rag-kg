from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.graphs import Neo4jGraph
import torch

class GraphBuilder:
    def __init__(self, model_name="google/flan-t5-small"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer,
                             device=0 if torch.cuda.is_available() else -1)

    def extract_relations(self, texts):
        graph_data = []
        for i, doc in enumerate(texts):
            text = doc.page_content.strip()
            if not text:
                continue
            result = self.pipe(f"Extract relationships: {text}", max_length=128, truncation=True)[0]['generated_text']
            graph_data.append({"page": i + 1, "content": result})
        return graph_data

    def store_graph(self, graph_data, neo4j_url, username, password):
        graph = Neo4jGraph(url=neo4j_url, username=username, password=password)
        graph.write_graph(graph_data)