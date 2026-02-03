import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import tempfile

# Set page title
st.set_page_config(page_title="Document Q&A System")

st.title("Document Question Answering (RAG)")

# ====================
# Step 0: Setup OpenAI API Key in Environment
# ====================
# Preferably set this in your OS environment or .env file safely
#openai.api_key = os.getenv("OPENAI_API_KEY") # Replace with your OpenAI API key
  # replace with your actual key or set externally

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("OpenAI API key not found in environment variables")
    st.stop()

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

# ====================
# Step 1: Load PDF document
# ====================
    #pdf_path = "/Users/abiramis/Downloads/transformer paper.pdf"  # replace with your actual PDF file path
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()  # List of Document objects
    

# ====================
# Step 2: Chunk the documents with overlap for better context
# ====================
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

# ====================
# Step 3: Generate embeddings with SentenceTransformer (all-MiniLM-L6-v2)
# ====================
    embeddings = OpenAIEmbeddings()

# ====================
# Step 4: Build FAISS vectorstore index to enable semantic search
# ====================
    vectorstore = FAISS.from_documents(texts, embeddings)

# ====================
# Step 5: Setup an advanced prompt template using Chain-of-Thought (CoT) style prompt engineering
# ====================
    prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a knowledgeable AI assistant that answers questions based on provided document context.
Think step-by-step and explain your reasoning as you go.

Document context:
{context}

Question:
{question}

Answer with detailed reasoning:
"""
)

# ====================
# Step 6: Initialize Chat Model (OpenAI GPT-3.5 Turbo) and Chain with prompt
# ====================
    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,max_tokens=1000)
    qa_chain = LLMChain(llm=chat_model, prompt=prompt_template)

# ====================
# Step 7: Ask a question and retrieve relevant document chunks
# ====================
    question =  st.text_input("Ask a question from the document")
    
    if question:
        # Retrieve top 3 relevant chunks based on semantic similarity
        retrieved_docs = vectorstore.similarity_search(question, k=3)

# Combine retrieved documents' text to form context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

# ====================
# Step 8: Run the Chain to generate an answer with step-by-step reasoning
# ====================
        answer = qa_chain.run({"context": context, "question": question})

        st.subheader("Answer")
        st.write(answer)
        st.write("PDF path on disk:", pdf_path)