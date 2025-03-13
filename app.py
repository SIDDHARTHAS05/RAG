# Streamlit App UI
import streamlit as st
from New_Final import process_pdf, generate_response
st.title("Financial RAG System")

# File Upload Section
uploaded_file = st.file_uploader("Upload Financial Statements (PDF)", type=["pdf"])
if uploaded_file:
    st.write("Processing PDF...")
    process_pdf(uploaded_file)
    st.success("PDF Processed Successfully!")

# Query Input
query = st.text_input("Enter your financial question:")
if query:
    response = generate_response(query)
    st.subheader("Response:")
    st.write(response)