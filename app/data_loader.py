from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from pathlib import Path
import fitz

def load_pdf(pdf_path: str):
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    return loader.load()

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_documents(documents, chunk_size=200, chunk_overlap=20):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)