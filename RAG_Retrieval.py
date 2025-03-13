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
