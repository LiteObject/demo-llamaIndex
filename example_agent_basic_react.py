from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# Set up the embedding model for converting text to vectors
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
# Set up the language model (LLM) for answering questions
Settings.llm = Ollama(model="phi4:latest",
                      temperature=0.1,
                      request_timeout=360.0)

llm = Settings.llm

def add(x: int, y: int) -> int:
    """Return the sum of x and y."""
    return x + y

def subtract(x: int, y: int) -> int:
    """Return the difference of x and y (x - y)."""
    return x - y

def multiply(x: int, y: int) -> int:
    """Return the product of x and y."""
    return x * y

def divide(x: int, y: int) -> int:
    """Return the quotient of x divided by y. Raises ValueError if y is zero."""
    if y == 0:
        raise ValueError("Cannot divide by zero.")
    return x / y

# Create tools from the defined functions
add_tool = FunctionTool.from_defaults(fn=add)
subtract_tool = FunctionTool.from_defaults(fn=subtract)
multiply_tool = FunctionTool.from_defaults(fn=multiply)
divide_tool = FunctionTool.from_defaults(fn=divide)

# Initialize the ReActAgent with the tools and language model
agent = ReActAgent.from_tools(
    [add_tool, subtract_tool, multiply_tool, divide_tool], llm=llm, verbose=True)

# Use the agent to answer a question
response = agent.chat("What is 20+(2*4)?")
print(response)
