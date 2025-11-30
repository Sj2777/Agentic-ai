from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()  # Load environment variables from .env file
import datetime
import os
import json
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_classic import hub
from langchain_core.prompts import ChatPromptTemplate 
from tavily import TavilyClient

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") #fetching the TAVILY_API_KEY from environment variables
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) #initializing the TavilyClient with the API key" we can use tavily_client to access Tavily's features

@tool
def tavily_search_tool(query:str, max_results:int=5)-> json:
    """This tool performs a search using the Tavily API and returns the results in json format."""
    
    response = tavily_client.search(query=query, max_results=max_results)
    return response.json()


@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S") :
    """Get the current system time in the specified format.""" #docstring for specifying the tool's purpose
   
    curr_time = datetime.datetime.now()
    formatted_time = curr_time.strftime(format) #.strftime to format the time
    return formatted_time

import requests

@tool
def get_weather_forecast(city: str, country: str = "IN") -> dict:
    """Get the current weather forecast for a given city."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},{country}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "city": city,
            "temperature": data["main"]["temp"],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"]
        }
    else:
        return {"error": f"Failed to fetch weather: {response.status_code}"}

llm=ChatGroq(model="llama-3.3-70b-versatile",)

react_prompt=hub.pull("hwchase17/react") #loading the react prompt template from hub

system_prompt_text = react_prompt.template#getting the system prompt text from the react prompt template

tools=[get_system_time, tavily_search_tool, get_weather_forecast]#defining the list of tools to be used by the agent

agent=create_agent(llm, tools, system_prompt=system_prompt_text) #creating the react agent with llm, tools and prompt template

# 🔄 Interactive loop until user says bye or exit
while True:
    query = input("User: \n")
    if query.lower() in ["bye", "exit", "stop"]:
        print("Agent: Goodbye! 👋")
        break

    result = agent.invoke({"messages": [HumanMessage(content=query)]}) #invoking the agent with the input query wrapped in HumanMessage
    #print (result)
    ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)] #filtering out the AI messages from the result messages

    # Format tool outputs nicely
    if "tool_calls" in result:
        for tool_call in result["tool_calls"]:
            output = tool_call["output"]
            if isinstance(output, dict) and "temperature" in output:
                print(f"Agent: Weather in {output['city']}: {output['temperature']}°C, "
                      f"{output['description']}, Humidity {output['humidity']}%, "
                      f"Wind {output['wind_speed']} m/s")
            elif output:
                print("Agent: \n", output)

    # Print only the final answer (no Thought/Action/Observation)
    if ai_messages:
        print("Final Answer:\n", ai_messages[-1].content)