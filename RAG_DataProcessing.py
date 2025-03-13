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