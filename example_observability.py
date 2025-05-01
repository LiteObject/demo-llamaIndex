import os

from dotenv import load_dotenv
from llama_index.core import (Settings, SimpleDirectoryReader, StorageContext,
                              VectorStoreIndex, load_index_from_storage,
                              set_global_handler)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# Load environment variables from .env file
load_dotenv()

# Get the Phoenix API key from environment variables
PHOENIX_API_KEY = os.getenv("PHOENIX_API_KEY")
# Set OpenTelemetry exporter headers for Phoenix
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={PHOENIX_API_KEY}"

# Set up the global handler for Arize Phoenix observability
set_global_handler(
    "arize_phoenix", endpoint="https://llamatrace.com/v1/traces")

# Configure embedding and LLM models using Ollama
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest")
Settings.llm = Ollama(model="phi4:latest",
                      temperature=0.1,
                      request_timeout=360.0)

# Load documents from the 'data' directory
documents = SimpleDirectoryReader("data").load_data()

# Check if a persistent storage directory exists
if os.path.exists("storage"):
    print("Loading index from storage...")
    # Load the index from existing storage
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    index = load_index_from_storage(storage_context)
else:
    print("Create the new index...")
    # Create a new index from documents and persist it
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir="storage")

# Create a query engine from the index
query_engine = index.as_query_engine()

# Query the index with a sample question
response = query_engine.query(
    "What is Byteville?")

# Print the response
print(response)
