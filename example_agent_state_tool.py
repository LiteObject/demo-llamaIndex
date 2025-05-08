from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.core.workflow import JsonSerializer

# Tool function to set the user's name in the agent's state
async def set_name(context: Context, name: str) -> str:
    """Set the name of the user."""

    # Print the function name and argument for debugging
    print("Invoked 'set_name' method with name:", name)
    # Retrieve the current state from the context (async)
    state = await context.get("state")
    # Update the state with the new name
    state["name"] = name
    # Save the updated state back to the context
    await context.set("state", state)
    return f"Name set to {name}"

# Initialize the Ollama LLM with a model that supports tools/function calling
llm = Ollama(model="llama3.2:latest",
             temperature=0.1,
             request_timeout=360.0)

# Create an agent workflow with the set_name tool and a helpful system prompt
workflow = AgentWorkflow.from_tools_or_functions(
    [set_name],
    llm=llm,
    system_prompt="You are a helpful assistant that can set a name."
)

# Create a context object to maintain state across agent interactions
ctx = Context(workflow)


async def main():
    # First interaction: introduce the user's name to the agent
    response = await workflow.run(user_msg="Hi, my name is Mohammed!", ctx=ctx)
    print(response)

    # Second interaction: ask the agent to recall the user's name using the same context
    response2 = await workflow.run(user_msg="What's my name?", ctx=ctx)
    print(response2)

    # Serialize the context to a dictionary using JSON (for saving or transferring state)
    ctx_dict = ctx.to_dict(serializer=JsonSerializer())

    # Restore a new context object from the serialized dictionary
    restored_ctx = Context.from_dict(
        workflow, ctx_dict, serializer=JsonSerializer()
    )

    # Third interaction: ask the agent to recall the user's name using the restored context
    response3 = await workflow.run(user_msg="What's my name?", ctx=restored_ctx)
    print(response3)

# Run the main function if this script is executed directly
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
