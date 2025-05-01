import logging
import sys

from llama_index.core import Settings, SQLDatabase
from llama_index.core.retrievers import NLSQLRetriever
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from sqlalchemy import create_engine

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest")
Settings.llm = Ollama(model="phi4:latest",
                      temperature=0.1,
                      request_timeout=360.0)

# Set up PostgreSQL connection
# Replace with your PostgreSQL credentials and database details
DB_USER = "user"
DB_PASSWORD = "password"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "city_db"

# Create SQLAlchemy engine for PostgreSQL
engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# Initialize SQLDatabase with the table we want to query
sql_database = SQLDatabase(engine, include_tables=[
                           "city_stats", "country_stats"])

# Configure Ollama LLM (ensure Ollama is running locally)
# llm = Ollama(model="llama3.2:latest", temperature=0.1, request_timeout=360.0)

# Create the NLSQLRetriever
nl_sql_retriever = NLSQLRetriever(
    sql_database, tables=["city_stats", "country_stats"], return_raw=True
)

# Perform a natural language query
results = nl_sql_retriever.retrieve(
    "Return the top 5 cities (along with their populations and countries) with the highest population."
)

# Display the result
for n in results:
    # display_source_node(n)
    print(n.get_text())
