# GreenLeaf Wellness Chatbot

An AI-powered customer support chatbot for GreenLeaf Wellness, a wellness and spa center in Italy.

## What it does

The chatbot answers customer questions based on the official FAQ documentation, using a RAG (Retrieval-Augmented Generation) architecture. It does not invent answers. It only responds using the provided knowledge base.

## Live Demo

The chatbot is deployed online and accessible here:

https://ciao-chatbot.onrender.com/docs

You can test the POST /ask endpoint directly from the browser.

Note: the service runs on Render Free tier. If it has been inactive for more than 15 minutes, the first request may take 1 to 2 minutes to wake it up.

## Tech Stack

- Python 3.12
- FastAPI for the web service
- LangChain for the RAG pipeline
- Cohere for the embeddings model embed-english-v3.0
- Groq API for the large language model openai/gpt-oss-120b
- ChromaDB as the vector database
- Uvicorn as the ASGI server

## How it works

1. The documents in the docs folder are loaded and split into chunks.
2. Each chunk is converted into a vector using Cohere embeddings.
3. The vectors are stored in ChromaDB.
4. When a user asks a question, the system finds the most relevant chunks.
5. These chunks are passed to the LLM together with a system prompt.
6. The LLM generates an answer based only on the provided context.

## Features

- Retrieval-Augmented Generation (RAG) for accurate answers
- Citation enforcement: the chatbot must cite the source document for every statement
- FastAPI web service with interactive documentation at /docs
- Cloud deployment on Render
- Environment variables for API keys (no secrets in the code)

## API Endpoints

- POST /ask : accepts a JSON body with a question field and returns the chatbot answer
- GET /health : returns the service status, useful for monitoring


## Project Structure
ciao/
docs/
faq.txt the knowledge base documents
api.py the FastAPI web service (used online on Render)
main.py the terminal chatbot (used locally on your PC)
requirements.txt the list of Python libraries
.env your API keys (never uploaded to GitHub)
.gitignore files to exclude from Git
README.md this file

## Setup

## How it works

1. The documents in the docs folder are loaded and split into chunks.
2. Each chunk is converted into a vector using Cohere embeddings.
3. The vectors are stored in ChromaDB.
4. When a user asks a question, the system finds the most relevant chunks.
5. These chunks are passed to the LLM together with a system prompt.
6. The LLM generates an answer based only on the provided context.

## Features

- Retrieval-Augmented Generation (RAG) for accurate answers
- Citation enforcement: the chatbot must cite the source document for every statement
- FastAPI web service with interactive documentation at /docs
- Cloud deployment on Render
- Environment variables for API keys, no secrets in the code

## API Endpoints

- POST /ask : accepts a JSON body with a question field and returns the chatbot answer
- GET /health : returns the service status, useful for monitoring

## Setup

Follow these steps to run the project on your computer.

1. Clone the repository:

   git clone https://github.com/petersmuditha/ciao.git
   cd ciao

2. Create a virtual environment:

   python -m venv venv

3. Activate the virtual environment:

   Windows:
   .\venv\Scripts\activate

   macOS and Linux:
   source venv/bin/activate

4. Install the dependencies:

   pip install -r requirements.txt

5. Create a .env file in the project root with your API keys:

   OPENAI_API_KEY=gsk_your_groq_key_here
   COHERE_API_KEY=your_cohere_key_here

   Get a free Groq key at https://console.groq.com
   Get a free Cohere key at https://dashboard.cohere.com

6. Make sure your documents are inside the docs folder as .txt files.

7. Run the web service locally:

   uvicorn api:app --reload --host 0.0.0.0 --port 8000

8. Open the browser at:

   http://127.0.0.1:8000/docs

   Use the POST /ask endpoint to ask a question.

## Running the Terminal Version

If you prefer to chat with the bot directly from the terminal, use main.py instead:

   python main.py

Type exit to quit.

## Example Questions

- How much does a relaxing massage cost
- How can I book an appointment
- What is your cancellation policy
- Do you offer gift cards
- What should I bring to my appointment

## Deployment

The project is deployed on Render.

Build Command:

   pip install -r requirements.txt

Start Command:

   uvicorn api:app --host 0.0.0.0 --port $PORT

Environment Variables required on Render:

- OPENAI_API_KEY
- COHERE_API_KEY

## Author

Muditha Peters
GitHub: https://github.com/petersmuditha
Email: petersmuditha@gmail.com

