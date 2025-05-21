import logging
import sys

from llama_index.core import Settings, SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import psycopg2
from sqlalchemy import create_engine
import sqlalchemy

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text:latest")
Settings.llm = Ollama(model="qwen2.5:7b-instruct-q8_0",
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

try:
    # Test the connection by opening a connection and closing it immediately
    with engine.connect() as connection:
        logging.info("Database connection successful.")
except (sqlalchemy.exc.OperationalError, psycopg2.OperationalError) as e:
    logging.error("Failed to connect to database: %s", e)
    sys.exit(1)

# Initialize SQLDatabase with the table we want to query
sql_database = SQLDatabase(engine, include_tables=[
                           "city_stats", "country_stats"])

# Configure Ollama LLM (ensure Ollama is running locally)
# llm = Ollama(model="llama3.2:latest", temperature=0.1, request_timeout=360.0)

# Create the NLSQLTableQueryEngine
query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    # Specify table to avoid context overflow
    tables=["city_stats", "country_stats"],
    # llm=llm,
    verbose=True
)

# Perform a natural language query
# QUERY_STR = "Which city has the lowest population?"
QUERY_STR = "Return the top 5 cities (along with their populations " \
    "and countries) with the highest population."
response = query_engine.query(QUERY_STR)
sql_query = response.metadata['sql_query']

# Display the result
print(f"\n>>> Query:\n{QUERY_STR}")
print(f"\n>>> SQL Query:\n{sql_query}")
print(f"\n>>> Response:\n{response}")
