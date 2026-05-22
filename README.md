<div align="center">

# Mental Health Chatbot Using Gen AI

### Compassionate AI-Powered Mental Health Support System

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logoColor=white)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logoColor=white)](https://groq.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)

</div>

---

## Overview

A conversational AI chatbot built with **Groq Cloud**, **LangChain**, and **HuggingFace embeddings** to provide mental health support through data-driven, compassionate responses. This system leverages RAG (Retrieval-Augmented Generation) to deliver evidence-based mental health information while maintaining a warm, empathetic conversational tone.

> **Disclaimer:** This chatbot is for informational and supportive purposes only. It is not a replacement for professional mental health care. If you are in crisis, please contact a licensed mental health professional or emergency services.

---

## Key Features

- **Empathetic Conversations** — LLM-powered responses tuned for compassionate mental health support
- **RAG Knowledge Base** — Evidence-based responses from curated mental health resources
- **Groq Ultra-Fast Inference** — Near-instantaneous responses for smooth conversation flow
- **HuggingFace Embeddings** — Semantic understanding of user concerns and emotions
- **Safe Response Guardrails** — Built-in safety mechanisms for crisis detection
- **Multi-Topic Support** — Anxiety, depression, stress, sleep issues, relationships, and more
- **Conversation Memory** — Context-aware responses that remember the conversation history

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logoColor=white)
![Groq](https://img.shields.io/badge/Groq_Cloud-F55036?style=for-the-badge&logoColor=white)
![Hugging Face](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?style=for-the-badge&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

---

## Architecture

```
User Message
    |
    v
Input Safety Check
    |
    v
HuggingFace Embedding Model
    |
    v
Vector Similarity Search (ChromaDB/Pinecone)
    |
    v
Context Retrieval from Mental Health Knowledge Base
    |
    v
Groq LLM (LangChain Chain)
    |
    v
Empathetic Response Generation
    |
    v
Crisis Detection Filter
    |
    v
Response to User
```

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/Avvv19/Mental-Health-Chatbot-Using-GEN-AI.git
cd Mental-Health-Chatbot-Using-GEN-AI

# Install dependencies
pip install -r requirements.txt

# Set API keys
export GROQ_API_KEY=your_groq_api_key

# Run the chatbot
jupyter notebook mental_health_chatbot.ipynb
```

---

## Topics Covered

- Anxiety and stress management
- Depression awareness and coping strategies
- Sleep hygiene and insomnia support
- Mindfulness and meditation guidance
- Relationship and communication support
- Work-life balance and burnout prevention
- Grief and loss processing
- Self-esteem and confidence building

---

## Author

**Venkata Vivek Varma Alluru** | AI in Healthcare

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/venkatavivekvarmaalluru/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Avvv19)
[![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@a.v.vivekvarma)
