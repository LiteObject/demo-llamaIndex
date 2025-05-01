# This script demonstrates using llama_index's AgentWorkflow with
# Ollama LLM, Yahoo Finance tools, and custom math tools.
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import FunctionAgent


def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b


tools = [
    # Register custom math tools. Add more tools here as needed.
    FunctionTool.from_defaults(multiply),
    FunctionTool.from_defaults(add)
]

# Initialize the Ollama LLM with a model that supports tools/function calling
llm = Ollama(model="llama3.2:latest",
             temperature=0.1,
             request_timeout=360.0)

# Create an agent workflow that can use the math tools
agent = FunctionAgent(
    name="math_agent",
    description="An agent that can perform basic mathematical operations using tools.",
    tools=tools,
    llm=llm,
    system_prompt="You are an agent that can perform basic mathematical operations using tools.",
    verbose=True)


async def main() -> None:
    """Main entry point for running the math agent demo."""
    try:
        # Ask the agent a math question using the custom math tools
        response = await agent.run(user_msg="What is 20+(2*4)?")
        print(response)
    except (RuntimeError, ValueError) as e:
        print(f"An error occurred while running the agent: {e}")

# Run the main function if this script is executed directly
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
