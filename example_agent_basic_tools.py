# This script demonstrates using llama_index's AgentWorkflow with
# Ollama LLM, Yahoo Finance tools, and custom math tools.
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec

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

# Create an agent workflow that can use the finance and math tools
workflow = AgentWorkflow.from_tools_or_functions(
    finance_tools,
    llm=llm,
    system_prompt="You are an agent that can perform basic mathematical operations using tools."
)

# Main async function to run the workflow with a finance question


async def main():
    # Ask the agent to get the current stock price of NVIDIA using the Yahoo Finance tool
    response = await workflow.run(user_msg="What's the current stock price of NVIDIA?")
    print(response)

# Run the main function if this script is executed directly
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
