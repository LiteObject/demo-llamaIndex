from sqlalchemy import create_engine
from llama_index.core import SQLDatabase, Settings
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest")
Settings.llm = Ollama(model="llama3.2:latest", request_timeout=360.0)

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
sql_database = SQLDatabase(engine, include_tables=["city_stats"])

# Configure Ollama LLM (ensure Ollama is running locally)
# llm = Ollama(model="llama3.2:latest", temperature=0.1, request_timeout=360.0)

# Create the NLSQLTableQueryEngine
query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["city_stats"],  # Specify table to avoid context overflow
    # llm=llm
)

# Perform a natural language query
QUERY_STR = "Which city has the lowest population?"
response = query_engine.query(QUERY_STR)

# Display the result
print(f"Query: {QUERY_STR}")
print(f"Response: {response}")
