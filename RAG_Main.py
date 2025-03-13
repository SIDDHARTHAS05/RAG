# Loading the model(RAG_Main.py)
import os
import pdfplumber
import chromadb
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load small open-source language model for response generation
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
lm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Initialize ChromaDB for vector storage
vector_db = chromadb.PersistentClient(path="./financial_db")
collection = vector_db.get_or_create_collection(name="financial_statements")

if __name__ == "__main__":
    print("Financial RAG System initialized.")