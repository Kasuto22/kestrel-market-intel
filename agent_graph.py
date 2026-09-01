from typing import TypedDict, List
from langgraph.graph import StateGraph, END
import yfinance as yf
import requests

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
    initial_state = {"commodity": "Natural Gas (US)"}
    final_state = energy_app.invoke(initial_state)

    print("\n-- Final Trading Report --")
    print(f"Commodity: {final_state['commodity']}")
    print(f"Price: ${final_state['current_price']}")
    print(f"Signal: {final_state['trade_signal']}")
    print(f"Reasoning: {final_state['reasoning']}")