CREATE TABLE IF NOT EXISTS transcripts (
        call_id VARCHAR(255) PRIMARY KEY,
        transcript LONGTEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

CREATE TABLE IF NOT EXISTS live_predictions (
        call_id VARCHAR(255) PRIMARY KEY,
        transcript LONGTEXT,
        prediction VARCHAR(50),
        confidence FLOAT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

 CREATE TABLE IF NOT EXISTS merged_sentiment_logs (
        call_id VARCHAR(255) PRIMARY KEY,
        student_name VARCHAR(255),
        tech_stack VARCHAR(255),
        location VARCHAR(255),
        remarks LONGTEXT,
        transcript_text LONGTEXT,
        cleaned_text LONGTEXT,
        label VARCHAR(50),
        sentiment VARCHAR(50),
        confidence FLOAT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );