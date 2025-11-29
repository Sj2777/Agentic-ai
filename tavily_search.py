from tavily import TavilyClient
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import os
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
response = tavily_client.search(query=query, max_results=max_results)

print(response)