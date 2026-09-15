import os
import sys
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from unittest.mock import MagicMock

# -- THE MONKEY PATCH --
# Ragas expects a VertexAI module that LangChain 0.3.x deleted. 
# We mock the module in memory so the import doesn't crash Python.
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()

# LangChain and Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

# Ragas Metrics and Evaluator
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
)

# Get existing vector database
from rag_tools import vectorstore

load_dotenv()

# Start Gemini LLM and Embeddings for ragas to use as a "judge"
eval_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
eval_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def generate_rag_answer(question: str, region: str):
    """Simulates RAG pipeline: Retreives context and generates an answer."""

    # Retreive context
    search_filter = {"region": region.upper()}
    docs = vectorstore.similarity_search(query=question, k=3, filter=search_filter)
    contexts = [doc.page_content for doc in docs]
    context_string = "\n\n".join(contexts)

    # Generate Answer
    prompt = f"""
    You are an expert energy regulatory analyst. Answer the question using ONLY the provided context.
    If the context does not contain the answer, say "I don't know".
    
    Context:
    {context_string}
    
    Question: {question}
    """
    response = eval_llm.invoke([HumanMessage(content=prompt)])

    # Extract text
    answer_text = response.content[0].get("text", "") if isinstance(response.content, list) else str(response.content)

    return answer_text.strip(), contexts

def main():
    print("-- Starting RAGAS Evaluation --")
    
    # Define the Golden Dataset
    # These are hard facts we know exist in the FERC and EU PDFs
    golden_data = [
        {
            "question": "What is the natural gas storage filling target for Member States for the year 2022?",
            "ground_truth": "For 2022, a lower filling target of 80% applies.",
            "region": "EU"
        },
        {
            "question": "Which specific section of the Natural Gas Act gives FERC exclusive authority to authorize applications for LNG import or export facilities?",
            "ground_truth": "FERC has exclusive authority under Section 3 of the NGA (15 U.S.C. § 717b).",
            "region": "US"
        },
        {
            "question": "What happens if a Member State does not have its own underground gas storage facilities?",
            "ground_truth": "They must demonstrate compliance with the burden-sharing mechanism and may partially comply by counting LNG stocks in existing floating storage units.",
            "region": "EU"
        }
    ]

    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    # Run questions through our pipeline
    print("Querying the vector database and generating answers...")
    for item in golden_data:
        ans, ctx = generate_rag_answer(item["question"], item["region"])
        
        questions.append(item["question"])
        answers.append(ans)
        contexts_list.append(ctx)
        ground_truths.append(item["ground_truth"])

    # Package the data for Ragas
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)

    # Run the Mathematical Evaluation
    print("Running Ragas LLM-as-a-judge evaluation...")
    result = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
        ],
        llm=eval_llm,
        embeddings=eval_embeddings
    )

    # Output the Scorecard
    print("\n-- RAGAS SCORECARD --")
    df = result.to_pandas()
    
    # Print the full evaluation table (metrics per question)
    print(df)
    
    # Automatically compute mean scores for all numeric metric columns
    print("\n--- AGGREGATE METRIC AVERAGES ---")
    numeric_cols = df.select_dtypes(include="number")
    if not numeric_cols.empty:
        print(numeric_cols.mean())
    else:
        print(result)

    print("\nDetailed breakdown saved to ragas_results.csv")
    df.to_csv("ragas_results.csv", index=False)

if __name__ == "__main__":
    main()