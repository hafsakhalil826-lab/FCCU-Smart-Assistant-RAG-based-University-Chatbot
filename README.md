# FCCU Smart Assistant — RAG-based University Chatbot

An AI-powered chatbot that answers questions from newly admitted FCCU students — covering admissions, programs, tuition fees, scholarships, hostel policy, and residential life.

Instead of relying only on the model's own knowledge, it uses **Retrieval-Augmented Generation (RAG)**: the bot retrieves the most relevant official FCCU documents first, then generates an accurate, grounded answer based on that content. This reduces hallucination and keeps responses factually tied to real university data.

## How it works

1. **Data Collection & Preprocessing** — Raw FCCU documents (admissions, programs, fees, scholarships, hostel/residential policies) are collected and cleaned into structured text files.
2. **Embedding & Vector Search** — Documents are embedded and stored in a **FAISS** vector index for fast semantic search.
3. **Retrieval-Augmented Generation** — On each query, the most relevant chunks are retrieved from FAISS and passed to an LLM (via **LangChain**) to generate a natural, context-aware answer.
4. **Interface** — A simple chat interface (HTML/CSS/JS) plus a GUI version in Python for local use.

## Tech Stack

- **Python** — core logic, data processing, RAG pipeline
- **LangChain** — orchestrating retrieval + generation
- **FAISS** — vector database for semantic search
- **Mistral LLM** — response generation
- **HTML / CSS / JavaScript** — chatbot frontend

## Use Case

Built for newly admitted FCCU students to instantly get accurate answers about:
- Admissions requirements & programs
- Tuition fees & scholarships
- Hostel policy & residential life
