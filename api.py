import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

load_dotenv()

app = FastAPI(title="GreenLeaf Wellness Chatbot API")

# Caricamento documenti e creazione database vettoriale
print("Loading documents...")
loader = DirectoryLoader('./docs/', glob="*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
documents = loader.load()

if len(documents) == 0:
    raise RuntimeError("No documents found in docs folder")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

for chunk in chunks:
    if "source" not in chunk.metadata:
        chunk.metadata["source"] = "unknown"
    else:
        chunk.metadata["source"] = os.path.basename(chunk.metadata["source"])

print("Creating vector database...")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

print("Loading reranking model...")
cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)

retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

template = """You are the virtual assistant of GreenLeaf Wellness a wellness and spa center in Italy.
Answer the user question based ONLY on the following context.
If the answer is not in the context say you do not know and invite the user to contact info at greenleafwellness dot it.

IMPORTANT CITATION RULES.
After every piece of information you provide you must cite the source document.
Use the format Source followed by the filename.
If you cannot cite a source for a statement do not include that statement.
Never invent or guess a source name.

Be friendly professional and helpful. Respond in the same language the user writes in.

Context
{context}

Question
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    temperature=0,
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

def format_docs_with_sources(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"Source {source} content {doc.page_content}")
    return "\n\n---\n\n".join(formatted)

rag_chain = (
    {
        "context": retriever | format_docs_with_sources,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Modello per la richiesta
class QuestionRequest(BaseModel):
    question: str

# Endpoint principale
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Risponde a una domanda usando il sistema RAG."""
    risposta = rag_chain.invoke(request.question)
    return {"answer": risposta}

# Endpoint di health check (utile per Render)
@app.get("/health")
async def health_check():
    return {"status": "ok"}