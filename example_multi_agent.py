import os
import datetime
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.core.agent.workflow import (
    AgentOutput,
    ToolCall,
    ToolCallResult,
)
from llama_index.core import Settings
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentWorkflow
from llm_factory import get_llm, get_embedding_model, LLMType

# Set up the embedding model and LLM for LlamaIndex
Settings.embed_model = get_embedding_model(llm_type=LLMType.OPENAI)
Settings.llm = get_llm(llm_type=LLMType.OPENAI)

# Initialize the Tavily tool for web search using API key from environment
# This tool will be used by the ResearchAgent to search the web
# and is converted to a tool list (only the first tool is used)
tavily_tool = TavilyToolSpec(api_key=os.getenv("TAVILY_API_KEY"))
search_web = tavily_tool.to_tool_list()[0]

# Tool function for recording research notes in the agent's state


async def record_notes(ctx: Context, notes: str, notes_title: str) -> str:
    """Useful for recording notes on a given topic."""
    current_state = await ctx.get("state")
    if "research_notes" not in current_state:
        current_state["research_notes"] = {}
    current_state["research_notes"][notes_title] = notes
    await ctx.set("state", current_state)
    return "Notes recorded."


# Tool function for writing a report and saving it in the agent's state
async def write_report(ctx: Context, report_content: str, filename: str = "report.md") -> str:
    """Useful for writing a report on a given topic."""
    current_state = await ctx.get("state")
    current_state["report_content"] = report_content
    await ctx.set("state", current_state)
    # Add datetime stamp to filename to avoid duplicates and save in docs folder
    base, ext = os.path.splitext(filename)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_with_stamp = f"{base}_{timestamp}{ext}"
    docs_path = os.path.join("./docs", filename_with_stamp)
    with open(docs_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    return f"Report written and saved to {docs_path}."


# Tool function for reviewing a report and saving feedback in the agent's state
async def review_report(ctx: Context, review: str) -> str:
    """Useful for reviewing a report and providing feedback."""
    current_state = await ctx.get("state")
    current_state["review"] = review
    await ctx.set("state", current_state)
    return "Report reviewed."


# Define the ResearchAgent: searches the web and records notes
research_agent = FunctionAgent(
    name="ResearchAgent",
    description="Useful for searching the web for information on a given topic and recording notes on the topic.",
    system_prompt=(
        "You are the ResearchAgent that can search the web for information on a given topic and record notes on the topic. "
        "Once notes are recorded and you are satisfied, you should hand off control to the WriteAgent to write a report on the topic."
    ),
    # llm=llm,
    tools=[search_web, record_notes],
    can_handoff_to=["WriteAgent"],
)

# Define the WriteAgent: writes a report based on research notes
write_agent = FunctionAgent(
    name="WriteAgent",
    description="Useful for writing a report on a given topic.",
    system_prompt=(
        "You are the WriteAgent that can write a report on a given topic. "
        "Your report should be in a markdown format. The content should be grounded in the research notes. "
        "Once the report is written, you should get feedback at least once from the ReviewAgent."
    ),
    # llm=llm,
    tools=[write_report],
    can_handoff_to=["ReviewAgent", "ResearchAgent"],
)

# Define the ReviewAgent: reviews the report and provides feedback
review_agent = FunctionAgent(
    name="ReviewAgent",
    description="Useful for reviewing a report and providing feedback.",
    system_prompt=(
        "You are the ReviewAgent that can review a report and provide feedback. "
        "Your feedback should either approve the current report or request changes for the WriteAgent to implement."
    ),
    # llm=llm,
    tools=[review_report],
    can_handoff_to=["WriteAgent"],
)

# Set up the agent workflow with all agents and initial state
agent_workflow = AgentWorkflow(
    agents=[research_agent, write_agent, review_agent],
    root_agent=research_agent.name,
    initial_state={
        "research_notes": {},
        "report_content": "Not written yet.",
        "review": "Review required.",
    },
)

# Main function to run the agent workflow and print outputs as events stream in


async def main():
    # Dynamically get current month and year
    current_month_year = datetime.datetime.now().strftime("%B %Y")
    handler = agent_workflow.run(user_msg=f"""
        Write a 5,000-word blog post that highlights and explains 5 to 7 of the most recent 
        and significant developments in health science as of {current_month_year}. Use the 
        most up-to-date information from trusted sources such as PubMed, ScienceDaily, Medical 
        News Today, Healthline, and the NIH. For each development, cite your sources in-line 
        in markdown format, including the article or study title, author(s) if available, 
        publication date, and a direct URL. Focus on topics that are relevant to general readers 
        and provide clear, accessible explanations of each breakthrough. Include relevant 
        statistics, cite the source and publication date of each study or article, and end with 
        a takeaway section summarizing why these updates matter for everyday health. Maintain a 
        tone that is informative yet conversational, suitable for a health-conscious audience 
        who may not have a medical background.
    """)

    current_agent = None
    current_tool_calls = ""
    async for event in handler.stream_events():
        # Print when the current agent changes
        if (
            hasattr(event, "current_agent_name")
            and event.current_agent_name != current_agent
        ):
            current_agent = event.current_agent_name
            print(f"\n{'='*50}")
            print(f"🤖 Agent: {current_agent}")
            print(f"{'='*50}\n")
        # Print agent output
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("📤 Output:", event.response.content)
            if event.tool_calls:
                print(
                    "🛠️  Planning to use tools:",
                    [call.tool_name for call in event.tool_calls],
                )
        # Print tool call results
        elif isinstance(event, ToolCallResult):
            print(f"🔧 Tool Result ({event.tool_name}):")
            print(f"  Arguments: {event.tool_kwargs}")
            print(f"  Output: {event.tool_output}")
        # Print when a tool is being called
        elif isinstance(event, ToolCall):
            print(f"🔨 Calling Tool: {event.tool_name}")
            print(f"  With arguments: {event.tool_kwargs}")

# Run the main function if this script is executed directly
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
