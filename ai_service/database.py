import os
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool

load_dotenv()

DB_URI = os.getenv("DATABASE_URL")

# Start connection pool for efficient database querying
pool = ConnectionPool(
    conninfo=DB_URI,
    max_size=20,
    kwargs={
        "autocommit": True,
        "prepare_threshold": 0,
    }
)

def get_pool():
    return pool