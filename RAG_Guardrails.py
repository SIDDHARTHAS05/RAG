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