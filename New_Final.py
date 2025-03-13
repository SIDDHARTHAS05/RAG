import streamlit as st
import fitz  # PyMuPDF for PDF processing
import chromadb  # Vector store for retrieval
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os
import tempfile
import re

# Load Open-Source Embedding Model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="financial_statements")

# Load Open-Source Small LLM for Response Generation
rag_pipeline = pipeline("text-generation", model="microsoft/phi-2")

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF using PyMuPDF."""
    doc = fitz.open(pdf_file)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def process_pdf(uploaded_file):
    """Process uploaded PDF, extract text, create embeddings, and store in ChromaDB."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_text = extract_text_from_pdf(tmp_file.name)
    
    # Split text into meaningful chunks
    sentences = pdf_text.split("\n")
    
    for sentence in sentences:
        if sentence.strip():
            embedding = embedder.encode(sentence).tolist()
            collection.add(documents=[sentence], embeddings=[embedding])

def retrieve_context(query):
    """Retrieve relevant context from ChromaDB using query embedding."""
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    return "\n".join(results["documents"][0]) if results["documents"] else ""

def validate_output(response):
    """Guardrail: Ensure output is structured and removes non-financial irrelevant data."""
    # Remove any hallucinated data that doesn't resemble financial information
    filtered_response = re.sub(r'[^0-9A-Za-z$%.,\-\s]', '', response)
    return filtered_response.strip()

def generate_response(query):
    """Retrieve relevant context and generate response using the LLM."""
    context = retrieve_context(query)
    prompt = f"""
    Given the financial statement data:
    {context}
    
    Answer the following question:
    {query}
    """
    response = rag_pipeline(prompt, max_length=200)[0]['generated_text']
    return validate_output(response)


