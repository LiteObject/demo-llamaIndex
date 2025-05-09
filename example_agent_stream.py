import os

from dotenv import load_dotenv
from llama_index.core.agent.workflow import (AgentInput, AgentOutput,
                                             AgentStream, AgentWorkflow,
                                             ToolCallResult)
from llama_index.core.workflow import Context
from llama_index.llms.ollama import Ollama
from llama_index.tools.tavily_research import TavilyToolSpec

# Load environment variables from .env file
load_dotenv()

# Initialize the Ollama LLM with a model that supports tools/function calling
llm = Ollama(model="llama3.2:latest",
             temperature=0.1,
             request_timeout=360.0)

# Ensure the TAVILY_API_KEY is set, raise an error if missing
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise EnvironmentError(
        "TAVILY_API_KEY is not set in the environment. Please add it to your .env file.")

# Create the Tavily search tool using the API key
tavily_tool = TavilyToolSpec(api_key=TAVILY_API_KEY)

# Create an agent workflow with the Tavily tool and a helpful system prompt
workflow = AgentWorkflow.from_tools_or_functions(
    tavily_tool.to_tool_list(),
    llm=llm,
    system_prompt="You're a helpful assistant that can search the web for information."
)


async def main():
    # Start the agent workflow with a user message and get a streaming handler
    handler = workflow.run(
        user_msg="What's the weather like in Frisco, TX?")

    # Handle streaming output from the agent
    async for event in handler.stream_events():
        if isinstance(event, AgentStream):
            # Print incremental output as the agent generates it
            print(event.delta, end="", flush=True)
        elif isinstance(event, AgentInput):
            # Print the current input messages and agent name
            print("Agent input: ", event.input)
            print("Agent name:", event.current_agent_name)
        elif isinstance(event, AgentOutput):
            # Print the full response, tool calls, and raw LLM response
            print("Agent output: ", event.response)
            print("Tool calls made: ", event.tool_calls)
            print("Raw LLM response: ", event.raw)
        elif isinstance(event, ToolCallResult):
            # Print details about the tool call and its output
            print("Tool called: ", event.tool_name)
            print("Arguments to the tool: ", event.tool_kwargs)
            print("Tool output: ", event.tool_output)

    # Print the final output after streaming is complete
    print(str(await handler))

# Run the main function if this script is executed directly
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
