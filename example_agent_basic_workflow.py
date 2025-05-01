# This script demonstrates using llama_index's AgentWorkflow with Ollama LLM and custom math tools.
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.ollama import Ollama

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

# Create an agent workflow that can use the add and multiply tools
workflow = AgentWorkflow.from_tools_or_functions(
    tools_or_functions=[multiply, add],
    llm=llm,
    system_prompt="You are an agent that can perform basic mathematical operations using tools.",
    verbose=True)

# Main async function to run the workflow with a math question
async def main():
    # Ask the agent to solve a math expression using the tools
    response = await workflow.run(user_msg="What is 20+(2*4)?")
    print(response)

# Run the main function if this script is executed directly
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
