import pandas as pd
from kafka import KafkaProducer
import json
import time

TOPIC_NAME = "happiness_features"
KAFKA_SERVER = "localhost:9092"

def json_serializer(data):
    return json.dumps(data).encode("utf-8")

if __name__ == "__main__":
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_SERVER],
        value_serializer=json_serializer
    )

    df = pd.read_csv("./data/processed/Happiness_Score.csv")

    features = ['gdp_per_capita', 'life_expectancy', 'social_support', 'score']
    df = df[features].dropna()

    print(f"Sending {len(df)} records to the topic '{TOPIC_NAME}'")

    for i, row in df.iterrows():
        data = {
            "gdp_per_capita": row["gdp_per_capita"],
            "life_expectancy": row["life_expectancy"],
            "social_support": row["social_support"],
            "score": row["score"] 
        }
        producer.send(TOPIC_NAME, value=data)
        print(f"{i+1} send")
        time.sleep(0.3) 

    producer.flush()
    print("Data sent successfully")