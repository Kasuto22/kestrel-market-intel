import os
import time
import urllib.parse
from typing import TypedDict, List
from dotenv import load_dotenv

import feedparser
import requests
import yfinance as yf
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver
from database import pool
from rag_tools import lookup_regulatory_policy


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
    location: str
    current_price: float
    price_status: str
    weather_summary: str
    news_headlines: List[str]
    news_sentiment: str
    regulatory_context: str
    trade_signal: str
    reasoning: str

@tool
def fetch_news_headlines(query: str) -> str:
    """Use this tool to search Google News for real-time headlines. 
    Pass a specific search query like 'US natural gas Henry Hub outages' or 'European natural gas TTF storage'.
    """
    encoded_query = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

    try:
        feed = feedparser.parse(rss_url)
        if not feed.entries:
            return "No news found for this query. Try a different search term."

        headlines = [f"- {entry.title}" for entry in feed.entries[:4]]
        return "\n".join(headlines)

    except Exception as e:
        return f"Tool Error: {e}"

# Prompt for the autonomous agent
news_agent_prompt = """You are an expert energy market analyst.
1. Use fetch_news_headlines to gather recent events for the requested commodity.
2. Use lookup_regulatory_policy with the appropriate region ('EU' or 'US') to verify any storage mandates, infrastructure constraints, or regulatory rules affecting supply and demand.
3. Synthesize your findings into an overall market sentiment: BULLISH, BEARISH, or NEUTRAL.

Format your final response EXACTLY as a strict string separated by two pipe characters:
SENTIMENT || SUMMARY || REGULATORY_NOTE
Example: BULLISH || Escalating supply risks in Norway || Under REPowerEU mandates, storage targets require accelerated injection schedules.
"""

# ReAct agent
news_agent = create_agent(
    model=llm,
    tools=[fetch_news_headlines, lookup_regulatory_policy],
    system_prompt=news_agent_prompt
)


# The Agents
def data_fetcher_node(state: MarketState):
    print("-- Fetching Market Data --")

    commodity = state.get("commodity", "")
    user_location = state.get("location", "").strip()

    # Determine region defaults
    is_eu = "EU" in commodity
    ticker = "TTF=F" if is_eu else "NG=F"
    temp_unit = "°C" if is_eu else "°F"

    # Resolve Location Coordinates
    target_location = user_location if user_location else ("Berlin" if is_eu else "Houston")
    lat, lon = None, None
    resolved_name = target_location

    try:
        geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={target_location}&count=1&format=json"
        geo_resp = requests.get(geocode_url, timeout=5)
        
        if geo_resp.status_code == 200:
            geo_data = geo_resp.json()
            if "results" in geo_data and len(geo_data["results"]) > 0:
                top_match = geo_data["results"][0]
                lat = top_match["latitude"]
                lon = top_match["longitude"]
                resolved_name = f"{top_match.get('name', target_location)}, {top_match.get('country_code', '')}"
                print(f"Data Fetcher: Geocoded '{target_location}' to {resolved_name} ({lat}, {lon})")
            else:
                print(f"Data Fetcher: Location '{target_location}' not found. Using regional fallback.")
        else:
            print(f"Data Fetcher: Geocoding API returned status code {geo_resp.status_code}")
    except Exception as e:
        print(f"Data Fetcher Error (Geocoding): {e}")

    # Fallbacks if geocoding returns no match or fails
    if lat is None or lon is None:
        if is_eu:
            lat, lon = 52.52, 13.41
            resolved_name = "Berlin (EU Fallback)"
        else:
            lat, lon = 29.76, -95.36
            resolved_name = "Houston (US Fallback)"

    # Fetch Market Price via yfinance
    current_price = None
    try:
        asset_data = yf.Ticker(ticker)
        recent_history = asset_data.history(period="5d")

        # If a 5-day window is empty, expand lookback
        if recent_history.empty:
            recent_history = asset_data.history(period="1mo")

        if not recent_history.empty:
            current_price = round(float(recent_history["Close"].iloc[-1]), 2)
            last_date = recent_history.index[-1].strftime("%Y-%m-%d")
            print(f"Data Fetcher: Retrieved {ticker} last close: ${current_price} ({last_date})")
        else:
            print(f"Data Fetcher Warning: No historical market data returned for {ticker}.")
    except Exception as e:
        print(f"Data Fetcher Error (Market): {e}")

    # Fallback to None rather than a synthetic price
    if current_price is None:
        price_status = "Market price currently unavailable."
    else:
        price_status = f"Last closing price: ${current_price}"

    # Fetch Dynamic Weather Data
    weather_unit_param = "" if is_eu else "&temperature_unit=fahrenheit"
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true{weather_unit_param}"
    
    weather_summary = f"Weather data unavailable for {resolved_name}."
    try:
        w_resp = requests.get(weather_url, timeout=5)
        if w_resp.status_code == 200:
            w_data = w_resp.json()
            temp = w_data["current_weather"]["temperature"]
            weather_summary = f"Current temperature in {resolved_name} is {temp}{temp_unit}."
            print(f"Data Fetcher: Weather retrieved - {weather_summary}")
        else:
            print(f"Data Fetcher: Weather API returned status code {w_resp.status_code}")
    except Exception as e:
        print(f"Data Fetcher Error (Weather): {e}")

    return {
        "current_price": current_price,
        "price_status": price_status,
        "weather_summary": weather_summary
    }

def news_analyst_node(state: MarketState):
    print("-- Analyzing Geopolitical News & Policy via Autonomous Sub-Agent --")
    commodity = state.get("commodity", "")
    region = "EU" if "EU" in commodity else "US"

    # Trigger autonomous agent loop
    inputs = {
        "messages": [
            HumanMessage(content=f"Analyze current news and regulatory policy for {commodity} in region {region}.")
        ]
    }

    try:
        # Agent loop (Reason -> Call Tool -> Observe -> Answer)
        result = news_agent.invoke(inputs)

        # Return the full conversation history. Final answer is the last message
        final_content = result["messages"][-1].content

        # Gemini's list format
        if isinstance(final_content, list):
            final_text = final_content[0].get("text", "")
        else:
            final_text = str(final_content)

        # Parse
        parts = final_text.strip().split("||")
        if len(parts) >= 3:
            sentiment = parts[0].strip()
            summary = parts[1].strip()
            regulatory_note = parts[2].strip()
        elif len(parts) == 2:
            sentiment = parts[0].strip()
            summary = parts[1].strip()
            regulatory_note = "No specific policy retrieved."
        else:
            sentiment = "NEUTRAL"
            summary = final_text
            regulatory_note = "Standard regulatory baseline."

    except Exception as e:
        print(f"News/Policy Agent Error: {e}")
        sentiment = "NEUTRAL"
        summary = "Failed to run autonomous analysis."
        regulatory_note = "Policy lookup failed."

    return {
        "news_headlines": [summary],
        "news_sentiment": sentiment,
        "regulatory_context": regulatory_note
    }

    

def supervisor_node(state: MarketState):
    print("-- Supervisor Synthesizing Signal --")

    price = state.get("current_price")
    weather = state.get("weather_summary", "")
    commodity = state.get("commodity", "")
    news_sentiment = state.get("news_sentiment", "NEUTRAL")
    headlines = state.get("news_headlines", [])
    headlines_text = "\n".join(headlines[:2])

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
    user_data = f"""
    Commodity: {state.get('commodity')}
    Location: {state.get('location')}
    Price Status: {state.get('price_status')}
    Weather Context: {state.get('weather_summary')}
    News Sentiment: {state.get('news_sentiment')}
    Headlines: {state.get('news_headlines')}
    Regulatory Constraints: {state.get('regulatory_context')}
    """

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

# Create database memory and tables automatically
saver = PostgresSaver(pool)
saver.setup()

# Compile with checkpointer attached
energy_app = workflow.compile(checkpointer=saver)

# Test
if __name__ == "__main__":
    # Unique ID for this run
    config = {"configurable": {"thread_id": "trading_desk_session_01"}}

    initial_state = {"commodity": "Natural Gas (EU)"}

    # Pass config with initial state
    final_state = energy_app.invoke(initial_state, config=config)

    print("\n-- Final Trading Report --")
    print(f"Commodity: {final_state['commodity']}")
    print(f"Price: ${final_state['current_price']}")
    print(f"Signal: {final_state['trade_signal']}")
    print(f"Reasoning: {final_state['reasoning']}")
    print("Flushing traces to LangSmith...")
    time.sleep(3) # Wait 3 seconds before killing the script
    # Cleanly shut down the database background threads
    pool.close()