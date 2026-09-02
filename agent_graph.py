import os
import time
from dotenv import load_dotenv

load_dotenv()
print("Is Tracing On?", os.getenv("LANGCHAIN_TRACING_V2"))

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import yfinance as yf
import requests

# Get API key
load_dotenv()

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash",
    temperature = 0.1, # Low temp for analytical, predictable answers
)

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

    # Read commodity from the State to get correct ticker
    commodity = state.get("commodity", "")

    # Dynamic Routing based on Market Region
    if "EU" in commodity:
        ticker = "TTF=F"
        lat, lon = "52.52", "13.41" # Berlin, Germany
        region_name = "Berlin (EU Proxy)"
        temp_unit = "°C"
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    else:
        ticker = "NG=F"
        lat, lon = "29.76", "-95.36" # Houston, TX (US Proxy)
        region_name = "Houston (US Proxy)"
        temp_unit = "°F"
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&temperature_unit=fahrenheit"

    try:
        # Fetch most recent trading day's data
        asset_data = yf.Ticker(ticker)
        recent_history = asset_data.history(period="1d")

        if not recent_history.empty:
            # Get closing price and round to 2 decimals
            current_price = round(float(recent_history["Close"].iloc[-1]), 2)
            print(f"Data Fetcher: Succesfully retrieved {ticker} closing price at {current_price}.")
        else:
            current_price = 0.00
            print("Data Fetcher: Market data unavailable.")
    except Exception as e:
        print(f"Data Fetcher Error: {e}")
        current_price = 0.00

    # Get Dynamic Weather Data
    weather_summary = "Weather data unavailable."
    try:
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            temp = data["current_weather"]["temperature"]
            weather_summary = f"Current temperature in {region_name} is {temp}{temp_unit}."
            print(f"Data Fetcher: Weather retrieved - {weather_summary}")
        else:
            print(f"Data Fetcher: Weather API returned status code {response.status_code}")
    except Exception as e:
        print(f"Data Fetcher Error (Weather): {e}")

    return {
        "current_price": current_price,
        "weather_summary": weather_summary
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
    commodity = state.get("commodity", "")

    # System Prompt
    system_instruction = """ 
    You are a senior quantitative energy trader analyzing natural gas markets.
    Evaluate the provided market data and determine a trading signal: BULLISH, BEARISH, or NEUTRAL.

    Energy Trading Rules for Weather (The U-Shaped Demand Curve):
    1. Extreme Cold (Below 45°F / 7°C):
       - High space-heating demand draws down gas storage rapidly.
       - Market Impact: BULLISH.
    2. Extreme Heat (Above 82°F / 28°C):
       - Peak air-conditioning load forces gas-fired power plants (power burn) to run at capacity.
       - Market Impact: BULLISH.
    3. Mild / Shoulder Season (55°F - 75°F / 13°C - 24°C):
       - Low heating and cooling load allows gas inventories to build.
       - Market Impact: BEARISH.
    4. Transition / Normal Range (45°F - 54°F or 76°F - 81°F):
       - Demand remains near seasonal baselines.
       - Market Impact: NEUTRAL.

    Format your response EXACTLY as a strict string separated by a pipe character:
    SIGNAL | REASONING
    Example: BULLISH | Temperatures exceeding 90°F in Texas drive significant power burn for air conditioning.
    """

    # Package data for LLM
    user_data = f"Commodity: {commodity}\nCurrent Price: {price}\nWeather: {weather}"

    # LLM call
    messages = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=user_data)
    ]

    try:
        response = llm.invoke(messages)

        # Extract text
        if isinstance(response.content, list):
            result_text = response.content[0].get("text", "")
        else:
            result_text = response.content

        result_text = str(result_text).strip()

        # Parse output into LangGraph state
        parts = result_text.split("|")
        if len(parts) == 2:
            signal = parts[0].strip()
            reasoning = parts[1].strip()
        else:
            signal = "UNKNOWN"
            reasoning = f"Failed to parse LLM output: {result_text}"

    except Exception as e:
        print(f"LLM Error: {e}")
        signal = "ERROR"
        reasoning = "The LLM failed to generate a response."

    return {
        "trade_signal": signal,
        "reasoning": reasoning
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
    initial_state = {"commodity": "Natural Gas (US)"}
    final_state = energy_app.invoke(initial_state)

    print("\n-- Final Trading Report --")
    print(f"Commodity: {final_state['commodity']}")
    print(f"Price: ${final_state['current_price']}")
    print(f"Signal: {final_state['trade_signal']}")
    print(f"Reasoning: {final_state['reasoning']}")
    print("Flushing traces to LangSmith...")
    time.sleep(3) # Wait 3 seconds before killing the script