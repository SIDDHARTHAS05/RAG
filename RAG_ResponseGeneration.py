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