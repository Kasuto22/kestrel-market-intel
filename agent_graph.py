from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# Defined State
class MarketState(TypedDict):
    commodity: str
    current_price: float
    weather_summary: str
    news_headlines: List[str]
    trade_signal: str
    reasoning: str

# The Agents
def data_fetcher_node(state: MarketState):
    print("-- Fetching Market Data --")
    return {
        "current_price": 2.85,
        "weather_summary": "Unexpected freezing temperatures forecasted for Germany."
    }

def news_analyst_node(state: MarketState):
    print("-- Analyzing Geopolitical News --")
    return {
        "news_headlines": ["EU struggles to fill gas reserves before winter."]
    }

def supervisor_node(state: MarketState):
    print("-- Supervisor Synthesizing Signal --")
    price = state.get("current_price")
    weather = state.get("weather_summary", "")

    if "freezing" in weather.lower():
        signal = "BULLISH"
        reason = "Cold weather increases haeting demand, driving up prices."
    else:
        signal = "NEUTRAL"
        reason = "Market conditions stable."

    return {
        "trade_signal": signal,
        "reasoning": reason
    }

# The workflow Graph
workflow = StateGraph(MarketState)

workflow.add_node("Data_Fetcher", data_fetcher_node)
workflow.add_node("News_Analyst", news_analyst_node)
workflow.add_node("Supervisor", supervisor_node)

workflow.set_entry_point("Data_Fetcher")
workflow.add_edge("Data_Fetcher", "News_Analyst")
workflow.add_edge("News_Analyst", "Supervisor")
workflow.add_edge("Supervisor", END)

energy_app = workflow.compile()

# Test
if __name__ == "__main__":
    initial_state = {"commodity": "Natural Gas (EU)"}
    final_state = energy_app.invoke(initial_state)

    print("\n-- Final Trading Report --")
    print(f"Commodity: {final_state['commodity']}")
    print(f"Price: ${final_state['current_price']}")
    print(f"Signal: {final_state['trade_signal']}")
    print(f"Reasoning: {final_state['reasoning']}")