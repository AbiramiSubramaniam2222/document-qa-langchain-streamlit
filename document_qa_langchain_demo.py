import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(page_title="Document Q&A System")
st.title("📄 Document Question Answering (RAG)")

# -----------------------------
# OpenAI API Key check
# -----------------------------
if not os.getenv("OPENAI_API_KEY"):
    st.error("OpenAI API key not found in environment variables")
    st.stop()

# -----------------------------
# Upload PDF
# -----------------------------
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # -----------------------------
    # Load PDF
    # -----------------------------
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # -----------------------------
    # Split text
    # -----------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    # -----------------------------
    # Embeddings + Vector Store
    # -----------------------------
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # -----------------------------
    # Prompt
    # -----------------------------
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful AI assistant.
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    # -----------------------------
    # LLM
    # -----------------------------
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2
    )

    chain = prompt | llm

    # -----------------------------
    # Ask Question
    # -----------------------------
    question = st.text_input("Ask a question from the document")

    if question:
        docs = vectorstore.similarity_search(question, k=5)
        context = "\n\n".join([d.page_content for d in docs])

        response = chain.invoke(
            {"context": context, "question": question}
        )

        st.subheader("Answer")
        st.write(response.content)
    
        with st.expander(f"Retrieved Context ({len(docs)} chunks)"):
            for i, doc in enumerate(docs, 1):
                st.markdown(f"**Chunk {i}:**")
                st.write(doc.page_content[:800])