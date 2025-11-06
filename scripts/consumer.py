from kafka import KafkaConsumer
import json
import pickle
import pandas as pd
import mysql.connector
import numpy as np

TOPIC_NAME = "happiness_features"
KAFKA_SERVER = "localhost:9092"
MODEL_PATH = "./models/happiness_score_pca_model.pkl"

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
            family FLOAT,
            freedom FLOAT,
            life_expectancy FLOAT,
            social_support FLOAT,
            y_real FLOAT,
            y_pred FLOAT,
            is_train BOOLEAN
        );
    """)
    conn.commit()
    conn.close()

def insert_prediction(data, y_real, y_pred):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO happiness_predictions 
        (gdp_per_capita, family, freedom, life_expectancy, social_support, y_real, y_pred, is_train)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
    """, (
        data.get("gdp_per_capita"),
        data.get("family"),
        data.get("freedom"),
        data.get("life_expectancy"),
        data.get("social_support"),
        y_real,
        y_pred,
        data.get("is_train")
    ))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_database()
    create_table()

    with open(MODEL_PATH, "rb") as f:
        package = pickle.load(f)

    model = package["model"]
    scaler = package["scaler"]
    pca = package["pca"]
    features = package["features"]

    print("Model loaded correctly")
    print(f"Expected features: {features}")

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
        print(f"Message received: {data}")

        try:
            X_values = [data.get(feat) for feat in features]
            
            if any(x is None or (isinstance(x, float) and np.isnan(x)) for x in X_values):
                print(f"Skipping message with missing values: {X_values}")
                continue
            
            X_array = np.array(X_values).reshape(1, -1)
            
            X_model = pd.DataFrame([data], columns=features)
            X_scaled = scaler.transform(X_model)
            X_pca = pca.transform(X_scaled)
            
            y_real = data.get("score")
            y_pred = model.predict(X_pca)[0]
            
            insert_prediction(data, y_real, y_pred)
            
            print(f"✓ Prediction saved: real={y_real:.4f}, pred={y_pred:.4f}, is_train={data.get('is_train')}")
        
        except Exception as e:
            print(f" Error processing message: {e}")
            print(f"   Data received: {data}")
            continue