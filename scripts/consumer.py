from kafka import KafkaConsumer
import json
import pickle
import pandas as pd
import mysql.connector

TOPIC_NAME = "happiness_features"
KAFKA_SERVER = "localhost:9092"
MODEL_PATH = "./models/happiness_score_lr_model.pkl"

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "happiness_dw"
}

def create_database():
    conn = mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"]
    )
    cursor = conn.cursor()
    cursor.execute("CREATE DATABASE IF NOT EXISTS happiness_dw;")
    conn.close()

def create_table():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS happiness_predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            gdp_per_capita FLOAT,
            life_expectancy FLOAT,
            social_support FLOAT,
            y_real FLOAT,
            y_pred FLOAT
        );
    """)
    conn.commit()
    conn.close()

def insert_prediction(gdp, life_exp, social, y_real, y_pred):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO happiness_predictions 
        (gdp_per_capita, life_expectancy, social_support, y_real, y_pred)
        VALUES (%s, %s, %s, %s, %s);
    """, (gdp, life_exp, social, y_real, y_pred))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_database()
    create_table()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded correctly")

    consumer = KafkaConsumer(
        TOPIC_NAME,
        bootstrap_servers=[KAFKA_SERVER],
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="happiness-group"
    )

    print(f"Waiting for messages from the topic '{TOPIC_NAME}'.")

    for message in consumer:
        data = message.value
        print(f"Message received")

        X = pd.DataFrame([{
            "gdp_per_capita": data["gdp_per_capita"],
            "life_expectancy": data["life_expectancy"],
            "social_support": data["social_support"]
        }])

        y_real = data["score"]
        y_pred = model.predict(X)[0]

        insert_prediction(
            data["gdp_per_capita"],
            data["life_expectancy"],
            data["social_support"],
            y_real,
            y_pred
        )

        print(f"Predictions saved in local MySQL")