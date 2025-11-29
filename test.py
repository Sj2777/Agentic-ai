from dotenv import load_dotenv
from langchain_groq import ChatGroq 
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()  # Load environment variables from .env file

llm = ChatGroq(model = "llama-3.3-70b-versatile")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Write a poem about PICT in 4 lines.")
]

result = llm.invoke(messages)
print(result.content)


