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

llm=ChatGroq(model="llama-3.3-70b-versatile",)

react_prompt=hub.pull("hwchase17/react") #loading the react prompt template from hub

system_prompt_text = react_prompt.template#getting the system prompt text from the react prompt template

agent = create_agent(
    model=llm,
    tools=[get_system_time],
    system_prompt=system_prompt_text, 
)
query="Do a web search & tell me who won IPL 2025 final? Also give me today's date."

result = agent.invoke ({"messages" : [HumanMessage(content=query)]}) #invoking the agent with the input query wrapped in HumanMessage
#print (result)
ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)] #filtering out the AI messages from the result messages
tools=[get_system_time, tavily_search_tool]#defining the list of tools to be used by the agent

agent=create_agent(llm, tools) #creating the react agent with llm, tools and prompt template

for i, msg in enumerate(ai_messages, 1): #enumerating through the AI messages & numbering them starting from 1
    print(f"AI Message {i}:\n {msg.content}\n") #printing the AI messages in function if i want some dynamically changing acc to iterartor use f in print statement


if ai_messages:
    print("Final Answer:\n", ai_messages[-1].content) #printing the final answer from the last AI message




