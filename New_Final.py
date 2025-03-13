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

import os
import pdfplumber
try:
    from RAG_Main import collection, embedding_model
except ImportError:
    import sys
    sys.path.append(r'D:\BITS Pilani Sem 3\Assignment-All\Conv AI\Assignment-2\RAG_1\RAG_Main.py')
    from RAG_Main import collection, embedding_model

# Function to extract text from PDFs
def extract_text_from_pdfs(pdf_folder):
    documents = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            with pdfplumber.open(os.path.join(pdf_folder, pdf_file)) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                documents.append(text)
    return documents

# Function to embed and store documents
def embed_and_store_documents(documents):
    for i, doc in enumerate(documents):
        chunks = [doc[i:i+512] for i in range(0, len(doc), 512)]  # Chunking text
        embeddings = embedding_model.encode(chunks).tolist()
        for j, chunk in enumerate(chunks):
            collection.add(documents=[chunk], embeddings=[embeddings[j]], ids=[f"doc_{i}_chunk_{j}"])

if __name__ == "__main__":
    pdf_folder = r"D:\BITS Pilani Sem 3\Assignment-All\Conv AI\Assignment-2\RAG_1\data"  # Change this to your PDF folder
    docs = extract_text_from_pdfs(pdf_folder)
    embed_and_store_documents(docs)
    print("Data processed and stored successfully.")

from rank_bm25 import BM25Okapi
from RAG_Main import collection, embedding_model

def retrieve_documents(query):
    # Convert query into embedding
    query_embedding = embedding_model.encode(query, convert_to_tensor=True).cpu().tolist()

    # Fetch documents from ChromaDB
    search_results = collection.query(query_embeddings=[query_embedding], n_results=3)

    # Ensure the retrieved results are valid
    if "documents" not in search_results or not isinstance(search_results["documents"], list):
        print("Error: ChromaDB query result format is incorrect.")
        return []

    # Extract and flatten document list from search results
    retrieved_texts = [doc for sublist in search_results["documents"] for doc in sublist]

    # Tokenize retrieved texts for BM25 processing
    tokenized_texts = [doc.split() for doc in retrieved_texts]

    # Initialize BM25 with tokenized documents
    bm25 = BM25Okapi(tokenized_texts)
    bm25_scores = bm25.get_scores(query.split())

    print("BM25 Scores:", bm25_scores)

    # Re-rank retrieved documents based on BM25 scores
    reranked_results = sorted(zip(retrieved_texts, bm25_scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in reranked_results]  # Return only document texts

if __name__ == "__main__":
    test_query = "What was the net revenue of MSD in 2023?"
    retrieved_docs = retrieve_documents(test_query)
    print("Retrieved Documents:")
    for doc in retrieved_docs:
        print(doc)
# Response generation (RAG_ResponseGeneration.py)
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from RAG_Main import lm_model, tokenizer
from RAG_Retrieval import retrieve_documents

# Generate response using small open-source LLM
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from RAG_Main import lm_model, tokenizer

# Define device for GPU or CPU processing
device = "cuda" if torch.cuda.is_available() else "cpu"
lm_model.to(device)  # Move model to correct device

def generate_response(query, retrieved_docs):
    # Ensure retrieved_docs is a flat list of strings
    if not all(isinstance(doc, str) for doc in retrieved_docs):
        retrieved_docs = [str(doc) for sublist in retrieved_docs for doc in (sublist if isinstance(sublist, list) else [sublist])]

    if not retrieved_docs:
        return "No relevant financial information found."

    context = "\n".join(retrieved_docs)  # Now safely joins text strings

    input_text = f"Context: {context} \n Question: {query}"
    
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = lm_model.generate(**inputs, max_new_tokens=50)  # Limit response length for speed
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True) 
import re
from RAG_ResponseGeneration import generate_response
from RAG_Retrieval import retrieve_documents

# Input Guardrail: Filter irrelevant queries
def validate_query(query):
    financial_keywords = ["revenue", "profit", "loss", "earnings", "net income", "assets", "liabilities", "cash flow"]
    return any(keyword in query.lower() for keyword in financial_keywords)

# Output Guardrail: Remove unsupported claims
def filter_response(response):
    hallucination_patterns = [r"I think", r"It seems", r"Maybe", r"Possibly"]
    for pattern in hallucination_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            return "The system could not verify this answer with high confidence. Please refer to official reports."
    return response

if __name__ == "__main__":
    test_query = "What was the net revenue of MSD in 2023?"
    if validate_query(test_query):
        retrieved_docs = retrieve_documents(test_query)
        raw_response = generate_response(test_query, retrieved_docs)
        final_response = filter_response(raw_response)
        print("Final Answer:", final_response)
    else:
        print("Invalid financial question.")

import streamlit as st
from RAG_Retrieval import retrieve_documents
from RAG_ResponseGeneration import generate_response

# Streamlit UI with loading spinner
def main():
    st.title("📊 Financial RAG Q&A System")
    st.write("Ask questions about company financial statements.")

    user_query = st.text_input("🔎 Enter your financial question:")

    if st.button("Ask 💬"):
        if not user_query.strip():
            st.warning("⚠️ Please enter a valid financial question.")
        else:
            with st.spinner("🔍 Fetching answer... Please wait."):
                retrieved_docs = retrieve_documents(user_query)
                
                if not retrieved_docs:
                    st.error("❌ No relevant documents found. Try a different question.")
                    return
                
                response = generate_response(user_query, retrieved_docs)
            
            # Show Source Documents First
            st.subheader("📄 Source Documents:")
            for doc in retrieved_docs:
                if doc:  # Ensure doc is not None or empty
                    st.text_area("Document Snippet", doc, height=100)

            # Show AI Answer
            st.subheader("🤖 Answer:")
            st.write(response)

if __name__ == "__main__":
    main()
