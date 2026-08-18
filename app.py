import os
import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
import requests
import json
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda


# ============================================================
# 1. DEFINE TOOLS
# ============================================================

@tool
def search_movies(genre: str) -> str:
    """Search for Indian movies by genre."""

    movies = {
        "sci-fi": "Cargo, 2.0, Mr. India",
        "comedy": "3 Idiots, Hera Pheri, Munna Bhai M.B.B.S.",
        "action": "RRR, Vikram, Baahubali"
    }

    return movies.get(
        genre.lower(),
        "No movies found for that genre"
    )


@tool
def change__to_f(temp_c) -> str:
    """Converts Celsius temperature to Fahrenheit."""

    try:
        temp_c = float(temp_c)
        fahrenheit = temp_c * 1.8 + 32
        return str(fahrenheit)

    except (TypeError, ValueError):
        return "Invalid temperature input. Please enter a number in Celsius."


@tool
def get_weather(city: str) -> str:
    """Get current temperature for a given city name."""

    try:
        # Geocoding API
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"

        geo_params = {
            "name": city,
            "count": 1
        }

        geo_response = requests.get(
            geo_url,
            params=geo_params,
            timeout=10
        ).json()

        # Check whether city was found
        if "results" not in geo_response or not geo_response["results"]:
            return f"Could not find weather data for city: {city}"

        location = geo_response["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        # Weather API
        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code",
            "temperature_unit": "celsius"
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10
        ).json()

        # Check whether weather data was received
        if "current" not in weather_response:
            return f"Could not get current weather for {city}"

        current = weather_response["current"]

        result = {
            "resolved_city": location["name"],
            "temperature_celsius": current["temperature_2m"],
            "weather_code": current["weather_code"]
        }

        return json.dumps(result)

    except Exception:
        return f"Unable to get weather information for {city}"


tools = [
    get_weather,
    search_movies,
    change__to_f
]


# ============================================================
# 2. INITIALIZE MODEL & AGENT
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

llm_flash = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GEMINI_API_KEY,
    temperature=0
)


agent = create_agent(
    model=llm_flash,
    tools=tools,
    system_prompt=(
        "You are a specialized agent restricted ONLY to Indian weather and cinema. "
        "For any other roles, topics, questions, or general knowledge outside of "
        "Indian weather and movies, you must say exactly: "
        "'I am not authorized to answer questions outside of Indian weather and cinema.'"
    )
)


# ============================================================
# 3. INPUT MODEL
# ============================================================

class AgentInput(BaseModel):
    input: str = Field(
        description="Your message to the agent"
    )


# ============================================================
# 4. FORMAT INPUT
# ============================================================

def format_for_agent(x) -> dict:

    user_input = x["input"] if isinstance(x, dict) else x.input

    return {
        "messages": [
            ("user", user_input)
        ]
    }


# ============================================================
# 5. EXTRACT ONLY FINAL TEXT RESPONSE
# ============================================================

def extract_text_response(agent_output: dict) -> str:

    # If output is not a dictionary
    if not isinstance(agent_output, dict):
        return str(agent_output)

    # Get messages from the agent output
    messages = agent_output.get("messages")

    # Sometimes messages are nested
    if messages is None:

        for value in agent_output.values():

            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                break

    # No messages found
    if not messages:
        return str(agent_output)

    # Get the last message
    last = messages[-1]

    # Get content
    content = getattr(last, "content", str(last))

    # --------------------------------------------------------
    # CASE 1: Content is already a string
    # --------------------------------------------------------

    if isinstance(content, str):
        return content


    # --------------------------------------------------------
    # CASE 2: Content is a list
    # --------------------------------------------------------

    if isinstance(content, list):

        text_parts = []

        for item in content:

            # If item is a dictionary
            if isinstance(item, dict):

                # Only take actual text blocks
                if item.get("type") == "text":

                    text = item.get("text", "")

                    if text:
                        text_parts.append(text)

            # If item itself is a string
            elif isinstance(item, str):

                text_parts.append(item)

        return " ".join(text_parts).strip()


    # --------------------------------------------------------
    # CASE 3: Anything else
    # --------------------------------------------------------

    return str(content)


# ============================================================
# 6. CREATE AGENT CHAIN
# ============================================================

formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(
    input_type=AgentInput,
    output_type=str
)


# ============================================================
# 7. FASTAPI APP
# ============================================================

app = FastAPI(
    title="Indian Weather and Cinema Agent"
)


add_routes(
    app,
    formatted_agent_chain,
    path="/agent",
    playground_type="default"
)


# ============================================================
# 8. RUN SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 8000)
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
