import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load API key from .env file
load_dotenv()

print("Loading documents...")

# Load all .txt files from the docs folder
loader = DirectoryLoader('./docs/', glob="*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
documents = loader.load()

if len(documents) == 0:
    print("ERROR: No documents found in the docs folder.")
    exit()

print(f"Loaded {len(documents)} document(s).")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunk(s).")

# Add source metadata to each chunk for citation
for chunk in chunks:
    if "source" not in chunk.metadata:
        chunk.metadata["source"] = "unknown"
    else:
        # Keep only the filename, not the full path
        chunk.metadata["source"] = os.path.basename(chunk.metadata["source"])

# Create vector database with local HuggingFace embeddings
print("Creating vector database (this may take a few seconds)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("Vector database created successfully!")

# Base retriever (recovers the 3 most similar chunks)
# Since we removed the reranker, we reduce 'k' from 10 to 3
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# System prompt for LingoMagic with citation enforcement
template = """You are the virtual assistant of LingoMagic, an online language learning platform.
Answer the user's question based ONLY on the following context.
If the answer is not in the context, say you don't know and invite the user to contact support@lingomagic.com.

IMPORTANT - CITATION RULES:
- After every piece of information you provide, you MUST cite the source document.
- Use the exact format: [Source: <filename>]
- If you cannot cite a source for a statement, DO NOT include that statement.
- Never invent or guess a source name.

Be friendly, professional, and helpful. Respond in the same language the user writes in.

Context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM (Groq - free and OpenAI-compatible)
llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    temperature=0,
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Format documents with their sources visible to the LLM
def format_docs_with_sources(docs):
    """Format retrieved documents with their source labels visible to the LLM."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

# RAG chain with citation enforcement
rag_chain = (
    {
        "context": retriever | format_docs_with_sources,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Chat loop
print("\n--- LingoMagic Assistant ready! Type 'exit' to quit. ---\n")
while True:
    domanda = input("You: ")
    if domanda.lower() in ["esci", "exit", "quit"]:
        print("Goodbye!")
        break
    try:
        risposta = rag_chain.invoke(domanda)
        print(f"\nAssistant: {risposta}\n")
    except Exception as e:
        print(f"\nError: {e}\n")