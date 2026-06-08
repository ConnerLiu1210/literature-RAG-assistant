# 📄 Literature RAG Assistant

A Retrieval-Augmented Generation (RAG) application that allows users to upload one or multiple research papers, ask natural-language questions, and receive grounded answers with page-level citations.

---

## 🚀 Features

- 📚 Upload one or multiple PDF research papers
- 🔍 Semantic search using vector embeddings
- 🤖 Claude-powered question answering
- 📝 Page-level source citations
- 📊 Multi-document comparison mode
- 🏷 Automatic paper metadata extraction
- 🔄 Query rewriting for follow-up conversations
- 💬 Interactive chat interface built with Streamlit

---

## 🏗 System Architecture

PDF Upload
     │
     ▼
PDF Parsing
     │
     ▼
Text Chunking
     │
     ▼
Embedding Generation
     │
     ▼
Vector Store
     │
     ▼
Semantic Retrieval
     │
     ▼
Prompt Construction
     │
     ▼
Claude API
     │
     ▼
Grounded Answer + Citations
---

## ⚙️ Tech Stack

### Frontend
- Streamlit

### Backend
- Python

### AI / NLP
- Retrieval-Augmented Generation (RAG)
- Sentence Transformers
- Semantic Search
- Query Rewriting

### Vector Database
- FAISS

### LLM
- Claude (Azure Anthropic API)

### PDF Processing
- PyPDFLoader
- Custom Text Chunking Pipeline

---

## 📂 Project Structure

literature-rag-assistant/
│
├── data/
├── src/
│   ├── pdf_loader.py
│   ├── text_splitter.py
│   ├── vector_store.py
│   ├── prompt_builder.py
│   ├── llm_client.py
│   ├── query_rewriter.py
│   ├── metadata_extractor.py
│   └── rag_chain.py
│
├── streamlit_app.py
├── app.py
├── requirements.txt
└── README.md
---

## ✨ Core Pipeline

### 1. PDF Loading

Research papers are uploaded and parsed into page-level text.

### 2. Text Chunking

Documents are split into overlapping chunks to preserve context during retrieval.

### 3. Embedding Generation

Each chunk is converted into a dense vector representation using Sentence Transformers.

### 4. Vector Search

Relevant chunks are retrieved with semantic similarity search through FAISS.

### 5. Prompt Construction

Retrieved passages are combined into a structured prompt with source information.

### 6. Answer Generation

Claude generates a grounded answer based only on the retrieved evidence.

### 7. Source Attribution

The application displays page-level citations and supporting excerpts.

---

## 💡 Supported Use Cases

- Summarize research papers
- Compare multiple papers
- Identify datasets used in a study
- Extract methodology details
- Find experimental results
- Answer follow-up questions
- Retrieve author and publication information

---

## 🔧 Installation

Clone the repository:

bash git clone https://github.com/ConnerLiu1210/literature-rag-assistant.git cd literature-rag-assistant 

Install dependencies:

bash pip install -r requirements.txt 

Configure environment variables:

env AZURE_ANTHROPIC_API_KEY=your_api_key AZURE_ANTHROPIC_BASE_URL=your_endpoint AZURE_ANTHROPIC_MODEL=claude-sonnet-4-5 

Run the application:

bash streamlit run streamlit_app.py 

---

## 📸 Demo

1. Upload one or more PDF papers.
2. Ask a natural-language question.
3. Review the generated answer.
4. Inspect the retrieved sources and page-level citations.

---

## 📈 Future Improvements

- Hybrid retrieval (BM25 + Dense Retrieval)
- Cross-encoder reranking
- Persistent vector database
- Support for larger document collections
- Multi-modal document understanding
- Agentic workflow integration

---

## 👨‍💻 Author

Conner Liu

GitHub: https://github.com/ConnerLiu1210

LinkedIn: https://www.linkedin.com/in/conner-liu-652552326

---

## 📄 License

This project is licensed under the MIT Licens