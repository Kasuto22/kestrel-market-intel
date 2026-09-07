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
    news_sentiment: str
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
news_agent_prompt = """You are an expert energy news analyst. 
1. Use the fetch_news_headlines tool to gather recent news for the requested market. 
2. If the results aren't helpful, try the tool again with a different, more specific query.
3. Determine if the overall news sentiment is BULLISH, BEARISH, or NEUTRAL for natural gas prices.

Format your final response EXACTLY as a strict string separated by a pipe character:
SENTIMENT | One-sentence summary of the headlines
Example: BULLISH | Pipeline maintenance in Norway limits export capacity.
"""

# ReAct agent
news_agent = create_agent(
    model=llm,
    tools=[fetch_news_headlines],
    system_prompt=news_agent_prompt
)


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
        # Expand lookback to 5 days for weekends and market holidays
        asset_data = yf.Ticker(ticker)
        recent_history = asset_data.history(period="5d")

        if not recent_history.empty:
            # Grab the most recent closing price from the 5-day window
            current_price = round(float(recent_history["Close"].iloc[-1]), 2)
            print(f"Data Fetcher: Successfully retrieved {ticker} closing price at {current_price}.")
        else:
            # Graceful Fallback if Yahoo delists the ticker
            current_price = 71.50 if "EU" in commodity else 2.90
            print(f"Data Fetcher: Market data unavailable for {ticker}. Using fallback price: ${current_price}")
            
    except Exception as e:
        print(f"Data Fetcher Error: {e}")
        current_price = 71.50 if "EU" in commodity else 2.90

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
    print("-- Analyzing Geopolitical News via Autonomous Sub-Agent --")
    commodity = state.get("commodity", "")

    # Trigger autonomous agent loop
    inputs = {"messages": [HumanMessage(content=f"Find market news for {commodity} and determine the sentiment.")]}

    try:
        # Agent loop (Reason -> Call Tool -> Observe -> Answer)
        result = news_agent.invoke(inputs)

        # Return the full conversation history. Final answer is the last message
        final_content = result["messages"][-1].content

        # Gemini's list format
        if isinstance(final_content, list):
            final_text = final_content[0].get("text", "")
        else:
            final_text = final_content

        # Parse
        parts = str(final_text).strip().split("|")
        if len(parts) == 2:
            sentiment = parts[0].strip()
            summary = parts[1].strip()
        else:
            sentiment = "NEUTRAL"
            summary = final_text

    except Exception as e:
        print(f"News Agent Error: {e}")
        sentiment = "NEUTRAL"
        summary = "Failed to analyze news autonomously."

    return {
        "news_headlines": [summary],
        "news_sentiment": sentiment
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
    Commodity: {commodity}
    Current Price: ${price}
    Weather Demand Context: {weather}
    News Sentiment: {news_sentiment}
    Key Headlines:
    {headlines_text}
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