# Semantic Search RAG System (Python + LangChain)

A Retrieval-Augmented Generation (RAG) system that performs semantic search over PDF documents and uses Large Language Models (LLMs) to generate context-aware responses.

This project demonstrates how to build an end-to-end RAG pipeline using LangChain, vector embeddings, and local LLMs (Ollama).

The system loads documents, splits them into chunks, generates embeddings, stores them in a vector store, retrieves relevant context, and uses an LLM agent to answer user queries.

# Project Overview

Retrieval-Augmented Generation improves LLM responses by injecting relevant external knowledge during inference.

Instead of relying only on the model's training data, the system:

Retrieves relevant document chunks

Supplies them as context to the LLM

Generates accurate, grounded responses

# System Architecture
User Query
     │
     ▼
Query Embedding
     │
     ▼
Vector Similarity Search
     │
     ▼
Retrieve Relevant Chunks
     │
     ▼
LLM (Ollama / LangChain Agent)
     │
     ▼
Generated Response

# Features

# PDF Document Processing

Load PDF documents using LangChain's PyPDFLoader

Extract text automatically from files

# Document Chunking

Large documents are split into smaller pieces to improve retrieval accuracy.

Uses:

RecursiveCharacterTextSplitter

Example configuration:

Chunk Size: 500
Chunk Overlap: 50
# Semantic Search

Documents are converted into vector embeddings and stored in a vector database.

The system retrieves the most relevant chunks using similarity search.

Vector store used:

InMemoryVectorStore
# Retrieval-Augmented Generation (RAG)

The system retrieves document context and passes it to an LLM agent which generates a final answer.

This ensures responses are context-aware and grounded in the documents.

# Local LLM Support

This project uses Ollama models so it can run locally without external APIs.

Example model:

smollm2:135m
# Project Structure
semantic-search-rag/
│
├── rag.py
│   Main RAG pipeline using LangChain agents
│
├── semantic_search.py
│   Semantic search implementation using embeddings
│
├── Naveen_resume.pdf
├── Pan_card.pdf
├── Aadhar_Card.pdf
│   Example documents used for testing
│
├── requirements.txt
│   Project dependencies
│
└── README.md
# Installation
# 1 Clone Repository
git clone https://github.com/yourusername/semantic-search-rag.git

cd semantic-search-rag
# 2 Install Dependencies
pip install langchain
pip install langchain-community
pip install langchain-openai
pip install langchain-ollama
pip install langchain-text-splitters
pip install chromadb
pip install pypdf

Or use:

pip install -r requirements.txt
# How the Pipeline Works
# Step 1 — Load Documents

PDF files are loaded using:

PyPDFLoader

Example documents:

Resume

Identity documents

Project files

# Step 2 — Split Documents

Documents are divided into chunks to improve search accuracy.

RecursiveCharacterTextSplitter

This helps retrieve only the relevant part of a document instead of the entire file.

# Step 3 — Create Embeddings

Each chunk is converted into a vector representation.

Example embedding method used in this project:

DeterministicFakeEmbedding

(This is mainly for testing. Real projects use OpenAI, HuggingFace, or Ollama embeddings.)

# Step 4 — Store Embeddings

Vectors are stored inside a vector database.

InMemoryVectorStore

This enables fast similarity-based retrieval.

# Step 5 — Retrieve Relevant Context

When a user query is given:

The query is embedded

Vector similarity search retrieves relevant chunks

The retrieved chunks are returned as context

# Step 6 — LLM Response Generation

The LLM agent receives:

User query

Retrieved document context

Then generates a context-aware response.

# Example Usage
# Run Semantic Search
python semantic_search.py

Example query:

"A backend application written in Python that simulates a car showroom experience"

Output:

Score: 0.0021
Document: Resume section describing backend project
# Run RAG Agent
python rag.py

Example query:

can you explain about Google AI-ML Student Intern - Virtua?

Example output:

Assistant:
The Google AI-ML Student Intern program focuses on training students
in artificial intelligence and machine learning concepts while
working on real-world industry projects.
# Technologies Used
Technology	Purpose
Python	Core programming language
LangChain	RAG pipeline and agents
Ollama	Local LLM inference
Vector Embeddings	Semantic search
PyPDF	PDF document parsing
# Future Improvements

Planned improvements:

Use real embeddings (OpenAI / HuggingFace)

Add persistent vector databases (FAISS / Chroma)

Create Streamlit or FastAPI interface

Support multiple document types

Implement hybrid search (BM25 + vector search)

# License

This project is for educational and research purposes.
