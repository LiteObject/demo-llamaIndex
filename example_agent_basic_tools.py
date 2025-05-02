# This script demonstrates using llama_index's AgentWorkflow with
# Ollama LLM, Yahoo Finance tools, and custom math tools.
# For more tools: https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/tools
# or https://llamahub.ai/
import os

from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from llama_index.tools.tavily_research import TavilyToolSpec

load_dotenv()

# Define a tool to multiply two numbers


def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b

# Define a tool to add two numbers


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b


# Initialize the Ollama LLM with a model that supports tools/function calling
llm = Ollama(model="llama3.2:latest",
             temperature=0.1,
             request_timeout=360.0)

# Get Yahoo Finance tools and add custom math tools to the tool list
finance_tools = YahooFinanceToolSpec().to_tool_list()
finance_tools.extend([multiply, add])

# ddg_search_tools = DuckDuckGoSearchToolSpec().to_tool_list()

tavily_search_tools = TavilyToolSpec(
    api_key=os.environ['TAVILY_API_KEY']).to_tool_list()

# Create an agent workflow that can use the finance and math tools
workflow = FunctionAgent(
    name="general_agent",
    tools=finance_tools + tavily_search_tools,
    llm=llm,
    system_prompt=("You are an agent that can perform web searches "
                   "and provide information using tools."),
)

# Main async function to run the workflow with a finance question


async def main():
    # Ask the agent to get the current stock price of NVIDIA using the Yahoo Finance tool
    response = await workflow.run(user_msg="What's the current stock price of NVIDIA?")
    print(response, "\n")

    response = await workflow.run(user_msg="What are the latest international news?")
    print(response, "\n")

# Run the main function if this script is executed directly
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
