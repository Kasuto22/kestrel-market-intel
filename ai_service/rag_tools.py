import os 
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Start embeddings and vector store
embeddings =  GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
raw_uri = os.getenv("DATABASE_URL")
connection_string = raw_uri.replace("postgresql://", "postgresql+psycopg://")

vectorstore = PGVector(
    embeddings=embeddings,
    collection_name="energy_policies",
    connection=connection_string,
    use_jsonb=True,
)

@tool
def lookup_regulatory_policy(query: str, region: str) -> str:
    """Look up regulatory policies, market design rules, and compliance mandates.
    
    Args:
        query: The specific topic to search (e.g., 'gas storage requirements' or 'pipeline capacity rules').
        region: Must be either 'EU' or 'US'.
    """
    # Strict metadata filtering so regions don't cross contaminate
    search_filter = {"region": region.upper()}

    try:
        results = vectorstore.similarity_search(
            query=query,
            k=3,
            filter=search_filter
        )

        if not results:
            return f"No regulatory guidance found for region '{region}' on topic: {query}"

        formatted_results = []
        for idx, doc in enumerate(results, start=1):
            source = os.path.basename(doc.metadata.get("source", "Unknown Document"))
            formatted_results.append(f"[{idx}] Source: {source}\n{doc.page_content.strip()}")

        return "\n\n---\n\n".join(formatted_results)

    except Exception as e:
        return f"Tool Error: {str(e)}"