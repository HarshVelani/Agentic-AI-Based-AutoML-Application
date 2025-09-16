from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()  # Load environment variables from .env file

from langchain_core.prompts import ChatPromptTemplate


class LLMManager:
    """Manager for LLM interactions"""
    
    # def __init__(self):
    #     """Initialize LLMManager with Groq API key"""
    #     self.llm = ChatGroq(
    #          api_key= os.getenv("GROQ_API_KEY"),  # Fetching API key from environment variable
    #          model="llama-3.1-8b-instant",  # Using available model
    #          temperature=0.1
    #      )

    # openai/gpt-oss-120b
    # meta-llama/llama-4-scout-17b-16e-instruct
        
    def __init__(self):
        self.llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=os.getenv("GROQ_API_KEY"))

    def invoke(self, prompt: ChatPromptTemplate, **kwargs) -> str:
        messages = prompt.format_messages(**kwargs)
        response = self.llm.invoke(messages)
        return response.content
        