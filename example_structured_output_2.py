# Import necessary modules
from pydantic import BaseModel
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from llama_index.core.prompts import ChatPromptTemplate

#################################
# Structured output is important because it allows data to be easily parsed,
# validated, and used programmatically, reducing ambiguity and enabling automation.
#
# Instead of explicitly doing llm.as_structured_llm(...), every LLM class has
# a structured_predict function which allows you to more easily call the LLM with
# a prompt template + template variables to return a structured output in one line of code.
#################################

# Define a Pydantic model for structured output from the LLM


class Country(BaseModel):
    name: str  # Name of the country
    capital: str  # Capital city of the country
    languages: list[str]  # List of official languages


# Initialize the Ollama LLM with the specified model and parameters
llm = Ollama(
    model="llama3.2:latest",           # Use the latest version of the llama3.2 model
    temperature=0.1,                   # Low temperature for more deterministic output
    request_timeout=360.0              # Timeout in seconds for the request
)

# Create a chat prompt template with a user message
chat_prompt_tmpl = ChatPromptTemplate(
    message_templates=[
        ChatMessage.from_str(
            # User asks about a country
            "Tell me about the country {country_name}", role="user"
        )
    ]
)

# Call the LLM with the prompt and get a structured Country object as output
country = llm.structured_predict(
    Country,                   # The Pydantic model to parse the output into
    chat_prompt_tmpl,          # The chat prompt template
    country_name="Canada"      # The variable to fill in the prompt
)

# Print the resulting Country object
print(country)
