from typing import List
from ollama import ChatResponse
from pydantic import BaseModel, Field
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage

#################################
# Structured output is important because it allows data to be easily parsed,
# validated, and used programmatically, reducing ambiguity and enabling automation.
#################################

# Define a Pydantic model for structured output from the LLM


class Country(BaseModel):
    name: str  # Name of the country
    capital: str  # Capital city of the country
    languages: list[str]  # List of official languages


# Initialize the Ollama LLM with the specified model and parameters
llm = Ollama(model="llama3.2:latest",
             temperature=0.1,  # Low temperature for more deterministic output
             request_timeout=360.0)  # Timeout in seconds

# Convert the LLM to a structured LLM that outputs Country objects
sllm = llm.as_structured_llm(output_cls=Country)

# Create a chat message asking about Canada
input_msg = ChatMessage.from_str("Tell me about Canada.")

# Send the message to the LLM and get the structured response
output: ChatResponse = sllm.chat([input_msg])
output_obj: Country = output.raw  # The raw structured output as a Country object

# Print the string representation and the raw object
print(str(output))           # Shows the ChatResponse
print(type(output_obj))      # Shows <class '__main__.Country'>
print(output_obj.name)       # Correct: prints the country name
print(output_obj)            # Prints the Country object
