import mysql.connector
from mysql.connector import errorcode
from config.settings import DB_CONFIG
import pandas as pd
def connect_db():
    """Return a mysql.connector connection using DB_CONFIG."""
    try:
        conn = mysql.connector.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG.get("port", 3306),
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database=DB_CONFIG["database"],
            autocommit=True
        )
        return conn
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_BAD_DB_ERROR:
            # Database doesn't exist
            raise RuntimeError("Database not found. Create database 'banalytics' first or run the CREATE DB SQL.")
        else:
            raise


def ensure_tables_exist():
    """Create tables if they do not exist."""
    conn = mysql.connector.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG.get("port", 3306),
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        autocommit=True
    )
    cur = conn.cursor()
    # Create database if not exists
    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_CONFIG['database']}`;")
    cur.execute(f"USE `{DB_CONFIG['database']}`;")
    # transcripts
    cur.execute("""
    CREATE TABLE IF NOT EXISTS transcripts (
        id INT AUTO_INCREMENT PRIMARY KEY,
        call_id VARCHAR(255),
        transcript LONGTEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY(call_id)
    );
    """)
    # live_predictions
    cur.execute("""
    CREATE TABLE IF NOT EXISTS live_predictions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        call_id VARCHAR(255),
        transcript LONGTEXT,
        prediction VARCHAR(50),
        score FLOAT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)
    # call_logs (merged)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS call_logs (
        id INT AUTO_INCREMENT PRIMARY KEY,
        call_id VARCHAR(255),
        student_name VARCHAR(255),
        year VARCHAR(50),
        tech_stack VARCHAR(255),
        location VARCHAR(255),
        remarks LONGTEXT,
        transcript_text LONGTEXT,
        combined_text LONGTEXT,
        cleaned_text LONGTEXT,
        label VARCHAR(50),
        sentiment VARCHAR(50),
        sentiment_score FLOAT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY(call_id)
    );
    """)
    cur.close()
    conn.close()


# DB CRUD for transcripts & logs
def get_transcript_from_db(call_id: str):
    """Return transcript text or None."""
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT transcript FROM transcripts WHERE call_id=%s LIMIT 1", (call_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row[0] if row else None

def save_transcript_to_db(call_id: str, transcript_text: str):
    conn = connect_db()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO transcripts (call_id, transcript) VALUES (%s,%s) ON DUPLICATE KEY UPDATE transcript=%s, created_at=CURRENT_TIMESTAMP", (call_id, transcript_text, transcript_text))
        conn.commit()
    finally:
        cur.close()
        conn.close()

def save_live_prediction_db(call_id: str, transcript_text: str, prediction: str, score: float):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO live_predictions (call_id, transcript, prediction, score) VALUES (%s,%s,%s,%s)", (call_id, transcript_text, prediction, score))
    conn.commit()
    cur.close()
    conn.close()

def upsert_call_log(row: dict):
  
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO call_logs
        (call_id, student_name, year, tech_stack, location, remarks, transcript_text, combined_text, cleaned_text, label, sentiment, sentiment_score)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            student_name=VALUES(student_name),
            year=VALUES(year),
            tech_stack=VALUES(tech_stack),
            location=VALUES(location),
            remarks=VALUES(remarks),
            transcript_text=VALUES(transcript_text),
            combined_text=VALUES(combined_text),
            cleaned_text=VALUES(cleaned_text),
            label=VALUES(label),
            sentiment=VALUES(sentiment),
            sentiment_score=VALUES(sentiment_score),
            created_at=CURRENT_TIMESTAMP
    """, (
        row.get("call_id"),
        row.get("student_name"),
        row.get("year"),
        row.get("tech_stack"),
        row.get("location"),
        row.get("remarks"),
        row.get("transcript_text"),
        row.get("combined_text"),
        row.get("cleaned_text"),
        row.get("label"),
        row.get("sentiment"),
        row.get("sentiment_score")
    ))
    conn.commit()
    cur.close()
    conn.close()

def save_merged_sentiment_row(row):
    conn = connect_db()
    cur = conn.cursor()
    sql = """INSERT INTO merged_sentiment_logs
        (call_id, student_name, year, tech_stack, location, remarks, label, transcript_text, cleaned_text, sentiment, sentiment_score)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    cur.execute(sql, (
        row.get("call_id"),
        row.get("student_name"),
        row.get("year"),
        row.get("tech_stack"),
        row.get("location"),
        row.get("remarks"),
        row.get("label"),
        row.get("transcript_text"),
        row.get("cleaned_text"),
        row.get("sentiment"),
        row.get("sentiment_score"),
    ))
    conn.commit()
    cur.close()
    conn.close()

def get_all_transcripts_from_db():
    conn = connect_db()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT call_id, transcript AS transcript_text FROM transcripts ORDER BY created_at DESC;")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return pd.DataFrame(rows)