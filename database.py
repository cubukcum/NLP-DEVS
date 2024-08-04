# code to connect amazon rds postgresql database

import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def connect():
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASS')
    )
    return conn

# a basic query to test the connection
def test():
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM employees")
    print(cur.fetchone())
    cur.close()
    conn.close()

if __name__ == '__main__':
    test()

# End of file