import os
import re
import fitz  # PyMuPDF for text extraction
import pdfplumber  # For extracting tables
import faiss  # Vector database
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from transformers import pipeline  # Open-source SLM for response generation

nltk.download('punkt')
# Load the small SLM (Phi-2, only ~2.5GB RAM required)
generator = pipeline("text-generation", model="microsoft/phi-2", device="cpu")

# Function to extract raw text from a PDF file
def extract_text_from_pdf(file_path):
    """Extracts raw text from a PDF."""
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return text

# Function to extract tables from a PDF file and convert them to formatted text
def extract_tables_from_pdf(file_path):
    """Extracts tables from a PDF and converts them to formatted text."""
    tables_text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_table()
                if tables:
                    for table in tables:
                        tables_text += "\n".join([" | ".join(row) for row in table if row]) + "\n\n"
    except Exception as e:
        print(f"Error extracting tables from {file_path}: {e}")
    return tables_text

# Function to detect sections and subsections dynamically from the extracted text
def detect_sections_and_subsections(text):
    """Detects sections and subsections dynamically."""
    sections = {}
    current_section = "General Information"
    current_subsection = None
    sections[current_section] = {}

    # Regular expressions to identify sections and subsections
    section_pattern = re.compile(r"^(?:\d+\.\s*)?[A-Z][A-Za-z\s\-]+$")
    subsection_pattern = re.compile(r"^(?:\d+\.\d+\s*)?[A-Z][A-Za-z\s\-]+$")

    lines = text.split("\n")
    for line in lines:
        line = line.strip()

        # Check if the line matches the section pattern
        if re.match(section_pattern, line) and len(line) > 5:
            current_section = line.strip()
            sections[current_section] = {}
            current_subsection = None

        # Check if the line matches the subsection pattern
        elif re.match(subsection_pattern, line) and len(line) > 5:
            current_subsection = line.strip()
            sections[current_section][current_subsection] = ""

        # Otherwise, add the line to the current section/subsection
        else:
            if current_subsection:
                sections[current_section][current_subsection] += line + " "
            else:
                sections[current_section]["General"] = sections[current_section].get("General", "") + line + " "

    return sections

# Function to process all PDF reports in a folder
def process_reports_folder(folder_path):
    """Reads PDFs, extracts sections, subsections, and tables."""
    all_reports = {}

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing: {file_path}")

            # Extract text and tables from the PDF
            pdf_text = extract_text_from_pdf(file_path)
            pdf_tables = extract_tables_from_pdf(file_path)

            # Combine the extracted text and tables
            full_text = pdf_text + "\n\nExtracted Tables:\n" + pdf_tables
            structured_data = detect_sections_and_subsections(full_text)

            # Store the structured data in the dictionary
            all_reports[file_name] = structured_data

    return all_reports

# Define the folder containing the financial reports
reports_folder = "reports"

# Extract and combine text from all PDFs in the folder
structured_reports = process_reports_folder(reports_folder)

# Print the first few sections of the reports
n = 5
for report, content in structured_reports.items():
    print(f"\nReport: {report}")

    # Limit to first 5 sections
    section_count = 0
    for section, subsections in content.items():
        if section_count >= n:
            break  # Stop printing after 5 sections
        print(f"\n  Section: {section}")

        # Limit to first 5 subsections per section
        subsection_count = 0
        for subsection, text in subsections.items():
            if subsection_count >= n:
                break  # Stop printing after 5 subsections
            print(f"        Subsection: {subsection}")
            print(f"            {text[:300]}...")  # Print first 300 chars for preview
            subsection_count += 1

        section_count += 1


# This function recursively chunks the structured data while preserving section integrity.
# It ensures that each chunk does not exceed the maximum chunk size and maintains an overlap for continuity.
def recursive_chunking(structured_data, max_chunk_size=512, overlap=100):
    """Recursively chunks structured data while preserving section integrity."""
    chunks = []
    
    for section, subsections in structured_data.items():
        section_text = f"### {section} ###\n"
        
        for subsection, content in subsections.items():
            full_text = section_text + f"## {subsection} ##\n{content}"
            
            if len(full_text) <= max_chunk_size:
                chunks.append(full_text)
            else:
                sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', full_text)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= max_chunk_size:
                        current_chunk += sentence + " "
                    else:
                        chunks.append(current_chunk.strip())
                        overlap_text = " ".join(current_chunk.split()[-(overlap // 10):])
                        current_chunk = overlap_text + " " + sentence + " "

                if current_chunk:
                    chunks.append(current_chunk.strip())

    return chunks

# This function converts text chunks into embeddings using a pre-trained model.
def embed_text_chunks(text_chunks, model_name="all-MiniLM-L6-v2"):
    """Converts text chunks into embeddings."""
    model = SentenceTransformer(model_name)
    embeddings = np.array([model.encode(chunk) for chunk in text_chunks])
    return model, embeddings

# This function stores embeddings in FAISS for fast retrieval.
def store_embeddings_in_faiss(embeddings):
    """Stores embeddings in FAISS for fast retrieval."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# This function retrieves relevant text chunks for a given query using FAISS similarity search.
def retrieve_relevant_text(query, model, index, text_chunks, top_k=3):
    """Retrieves relevant text chunks for a given query using FAISS similarity search."""
    query_embedding = model.encode(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    retrieved_texts = [text_chunks[i] for i in indices[0]]
    return retrieved_texts

# This function generates a response using the Phi-2 (2.7B SLM) model.
def generate_response(query, retrieved_texts, generator):
    """Generates a response using Phi-2 (2.7B SLM)."""
    context = "\n\n".join(retrieved_texts)
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

    response = generator(prompt, max_length=3000, do_sample=True)[0]["generated_text"]    
    return response

# Combine all text chunks from the structured reports
all_text_chunks = []
for report, content in structured_reports.items():
    all_text_chunks.extend(recursive_chunking(content))

embedding_model, embeddings = embed_text_chunks(all_text_chunks)
faiss_index = store_embeddings_in_faiss(embeddings)

# Print Sample Output
print(f"Total Chunks: {len(all_text_chunks)}")
print(f"FAISS Index Size: {faiss_index.ntotal}")
# Example Query Retrieval
query = "What was Eros International’s revenue in FY24?"
retrieved_info = retrieve_relevant_text(query, embedding_model, faiss_index, all_text_chunks)

# Generate a response using the retrieved information
response = generate_response(query, retrieved_info, generator)
print("\nGenerated Answer using Simple RAG:")
print(response)

# Function to prepare a BM25 index for keyword-based retrieval
def prepare_bm25_index(text_chunks):
    """Prepares a BM25 index for keyword-based retrieval."""
    tokenized_corpus = [word_tokenize(chunk.lower()) for chunk in text_chunks]
    return BM25Okapi(tokenized_corpus)

# Function to perform multi-stage retrieval using BM25 and FAISS
def multi_stage_retrieval(query, model, faiss_index, bm25_index, text_chunks, top_k=5):
    """Retrieves relevant text chunks using BM25 + FAISS + Re-ranking."""
    # Encode the query using the embedding model
    query_embedding = model.encode(query).reshape(1, -1)
    
    # Retrieve top-k results using FAISS
    faiss_distances, faiss_indices = faiss_index.search(query_embedding, top_k)
    retrieved_faiss_texts = [text_chunks[i] for i in faiss_indices[0]]

    # Tokenize the query for BM25
    tokenized_query = word_tokenize(query.lower())
    
    # Retrieve top-k results using BM25
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_top_indices = np.argsort(bm25_scores)[-top_k:]
    retrieved_bm25_texts = [text_chunks[i] for i in bm25_top_indices]

    # Merge FAISS & BM25 results, removing duplicates
    combined_results = list(dict.fromkeys(retrieved_faiss_texts + retrieved_bm25_texts))

    return combined_results[:top_k]

# Prepare the BM25 index using all text chunks
bm25_index = prepare_bm25_index(all_text_chunks)

# Example query for multi-stage retrieval
query = "What was Eros International’s revenue in FY24?"
retrieved_info = multi_stage_retrieval(query, embedding_model, faiss_index, bm25_index, all_text_chunks)

# Generate a response using the retrieved information
response = generate_response(query, retrieved_info, generator)

# Print the AI-generated answer with BM25 indexing
print("\nGenerated Answer with bm25 indexing:")
print(response)