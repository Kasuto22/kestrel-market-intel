from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from agent_graph import energy_app

# Initialize FastAPI server
app = FastAPI(title="Kestrel Market Intel API")

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Expected JSON format
class MarketRequest(BaseModel):
    commodity: str
    thread_id: str
    user_prompt: Optional[str] = None
    location: Optional[str] = None

# POST endpoint
@app.post("/api/analyze")
async def analyze_market(query: MarketRequest):
    try:
        print(f"-- Received Request for {query.commodity} (Thread: {query.thread_id}) --")

        # LangGraph session memory
        config = {"configurable": {"thread_id": query.thread_id}}

        # Get frontend request into LangGraph state
        initial_state = {
            "commodity": query.commodity,
            "location": query.location or "Default Region"
        }

        # Multi-agent workflow
        final_state = energy_app.invoke(initial_state, config=config)

        # Return final state to frontend
        return {
            "status": "success",
            "thread_id": query.thread_id,
            "data": {
                "commodity": final_state.get("commodity"),
                "current_price": final_state.get("current_price"),
                "price_status": final_state.get("price_status"), 
                "weather_summary": final_state.get("weather_summary"),
                "news_sentiment": final_state.get("news_sentiment"),
                "news_headlines": final_state.get("news_headlines"), # <--- Add this line!
                "regulatory_context": final_state.get("regulatory_context"),
                "trade_signal": final_state.get("trade_signal"),
                "reasoning": final_state.get("reasoning")
            }
        }

    except Exception as e:
        print(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Kestrel AI Backend"}