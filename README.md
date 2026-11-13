
# ScamShield AI

### AI-Powered Scam Detection Using Retrieval-Augmented Generation (RAG)

ScamShield AI is an intelligent scam detection application that analyzes suspicious SMS messages, WhatsApp messages, and emails using Google's Gemini AI.

## Features
- Detects phishing and fraudulent messages.
- Identifies OTP and UPI payment scams.
- Detects fake job offers and investment scams.
- Classifies messages into Low, Medium, and High risk.
- Retrieves relevant fraud knowledge using FAISS.
- Generates AI-powered explanations and safety recommendations.

## Technology Stack
- Python
- Google Gemini API
- Retrieval-Augmented Generation (RAG)
- Sentence Transformers
- FAISS
- Streamlit

## Architecture
1. User submits a suspicious message.
2. Sentence Transformers generate embeddings.
3. FAISS retrieves relevant scam knowledge.
4. Gemini analyzes the message using retrieved context.
5. Streamlit displays the risk assessment.

## Project Structure
- app.py
- requirements.txt
- knowledge_base/
- .gitignore

## Run Locally

Install dependencies:

pip install -r requirements.txt

Configure your Gemini API key using Streamlit secrets or an environment variable.

Run:

streamlit run app.py

## Live Demo

https://scamshield--ai-drub8l4u5dq8y9zqebml.streamlit.app/

## Disclaimer
This project provides AI-assisted risk assessments and cannot guarantee detection of every fraudulent message.

## Author
AI Engineer Portfolio Project
