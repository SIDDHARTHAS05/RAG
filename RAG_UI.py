import streamlit as st
# from RAG_Retrieval import retrieve_documents
# from RAG_ResponseGeneration import generate_response

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
