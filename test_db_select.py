import os

import psycopg2
# Load environment variables from a .env file
from dotenv import load_dotenv

load_dotenv()

# Read DB_ values
db_host = os.getenv("DB_HOST")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_name = os.getenv("DB_NAME")
db_port = os.getenv("DB_PORT")

try:
    conn = psycopg2.connect(
        dbname=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port,
        connect_timeout=5
    )
    print("Connection successful!")

    # Create a cursor object using the connection
    cur = conn.cursor()

    # Execute the query to select all records from city_stats table
    cur.execute("SELECT * FROM city_stats;")

    # Fetch and print the results
    rows = cur.fetchall()
    for row in rows:
        print(row)
except (psycopg2.OperationalError, psycopg2.ProgrammingError) as e:
    print(f"Database connection error: {e}")
finally:
    if 'conn' in locals() and conn is not None:
        conn.close()
