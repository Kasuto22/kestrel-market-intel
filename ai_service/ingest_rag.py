import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

# Load env variables
load_dotenv()

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EU_DIR = os.path.join(BASE_DIR, "documents", "eu")
US_DIR = os.path.join(BASE_DIR, "documents", "us")

def main():
    print("-- Initializing RAG Ingestion Pipeline --")

    # Load documents and attach regional metadata
    print("Loading EU documents...")
    eu_loader = PyPDFDirectoryLoader(EU_DIR)
    eu_docs = eu_loader.load()
    for doc in eu_docs:
        doc.metadata["region"] = "EU"

    print("Loading US documents...")
    us_loader = PyPDFDirectoryLoader(US_DIR)
    us_docs = us_loader.load()
    for doc in us_docs:
        doc.metadata["region"] = "US"

    all_docs = eu_docs + us_docs
    print(f"Successfully loaded {len(all_docs)} total pages.")

    # Split into semantic chunks
    # Chunks of 1k characters with 100 character overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(all_docs)
    print(f"Sliced documents into {len(chunks)} embedded chunks.")

    # Start Google Embeddings and Database Connection
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Dynamically inject the psycopg3 driver into the SQLAlchemy connection string
    raw_uri = os.getenv("DATABASE_URL")
    connection_string = raw_uri.replace("postgresql://", "postgresql+psycopg://")
    
    print("Connecting to pgvector database...")
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="energy_policies",
        connection=connection_string,
        use_jsonb=True,
    )

    # Generate Embeddings and write to database
    print("Generating embeddings and writing to PostgreSQL...")
    vectorstore.add_documents(chunks)
    
    print("--- RAG Ingestion Complete ---")

if __name__ == "__main__":
    main()