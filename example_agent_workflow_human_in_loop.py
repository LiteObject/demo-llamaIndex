import asyncio
from dotenv import load_dotenv
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context, HumanResponseEvent, InputRequiredEvent
from llama_index.llms.ollama import Ollama

load_dotenv()

llm = Ollama(model="llama3.2:latest", temperature=0.1, request_timeout=360.0)


async def confirmed_task(ctx: Context) -> str:
    """
    A task that requires user confirmation before proceeding.
    """
    ctx.write_event_to_stream(
        InputRequiredEvent(
            prefix="Do you want to proceed with the task? ",
            user_name="Mohammed",
        )
    )
    response = await ctx.wait_for_event(
        HumanResponseEvent,
        requirements={"user_name": "Mohammed"}
    )
    if response.response.strip().lower() == "yes":
        return "Task completed successfully."
    else:
        return "Task aborted."

workflow = AgentWorkflow.from_tools_or_functions(
    [confirmed_task],
    llm=llm,
    system_prompt=("You are a helpful assistant that executes tasks "
                   "only with user confirmation and stops if the user declines."),
)


async def main() -> None:
    try:
        handler = workflow.run(user_msg="I want to proceed with the task.")
        async for event in handler.stream_events():
            if isinstance(event, InputRequiredEvent):
                try:
                    response = input(event.prefix)
                except (KeyboardInterrupt, EOFError):
                    print("\nInput cancelled by user. Aborting workflow.")
                    return

                handler.ctx.send_event(
                    HumanResponseEvent(response=response,
                                       user_name=event.user_name)
                )
        response = await handler
        print(f"Raw response: {response}")

    except Exception as e:
        print(f"Unexpected exception occurred: {type(e).__name__}: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
