# Document Question Answering System (RAG)
This project is a Document Question Answering system built using LangChain, FAISS, and Streamlit. Users can upload PDF documents and
ask natural language questions, which are answered using a Retrieval-Augmented Generation (RAG) pipeline.

**Tech stack**
- Python
- LangChain
- FAISS
- Sentence Transformers
- OpenAI API
- Streamlit
- AWS EC2 (Ubuntu)

## Architecture
1. PDF documents are loaded and split into chunks
2. Chunks are converted into vector embeddings
3. FAISS is used for semantic search
4. Relevant context is retrieved and passed to the LLM
5. Streamlit provides the web interface.

## Run Locally
pip install -r requirements.txt
streamlit run app.py

## Cloud Deployment
- Deployed on AWS EC2 (Ubuntu)
- Connected via SSH using key-based authentication
- Security groups configured for ports 22 (SSH) and 8501 (Streamlit)
- Application exposed via public EC2 IP

  ## Future Improvements
- Add authentication for multi-user access
- Deploy behind a reverse proxy (Nginx)
- Add logging and monitoring
- Support larger documents and batch uploads
