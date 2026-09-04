from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent_graph import energy_app

# Initialize FastAPI server
app = FastAPI(title="Kestrel Market Intel API")

# Expected JSON format
class MarketRequest(BaseModel):
    commodity: str

# POST endpoint
@app.post("/api/analyze")
async def analyze_market(request: MarketRequest):
    try:
        # Get frontend request into LangGraph state
        initial_state = {"commodity": request.commodity}

        # Multi-agent workflow
        final_state = energy_app.invoke(initial_state)

        # Return final state to frontend
        return {
            "commodity": final_state.get("commodity"),
            "current_price": final_state.get("current_price"),
            "weather_summary": final_state.get("weather_summary"),
            "news_headlines": final_state.get("news_headlines"),
            "news_sentiment": final_state.get("news_sentiment"),
            "trade_signal": final_state.get("trade_signal"),
            "reasoning": final_state.get("reasoning")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))