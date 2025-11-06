import pandas as pd
import numpy as np
import json
import time
from kafka import KafkaProducer
from sklearn.linear_model import LinearRegression
import pickle

TOPIC_NAME = "happiness_features"
KAFKA_SERVER = "localhost:9092"

# Load split indices
with open("./models/happiness_score_pca_model.pkl", "rb") as f:
    model_package = pickle.load(f)

# Extract
df_2015 = pd.read_csv("./data/processed/2015_with_flag.csv")
df_2016 = pd.read_csv("./data/processed/2016_with_flag.csv")
df_2017 = pd.read_csv("./data/processed/2017_with_flag.csv")
df_2018 = pd.read_csv("./data/processed/2018_with_flag.csv")
df_2019 = pd.read_csv("./data/processed/2019_with_flag.csv")
df = [df_2015, df_2016, df_2017, df_2018, df_2019]

for i, year in zip(df, range(2015, 2020)):
    i["year"] = year

# Transform
df_2019.rename(columns={
    "Country or region": "country_or_region",
    "Overall rank": "rank",
    "Score": "score",
    "Social support": "social_support",
    "GDP per capita": "gdp_per_capita",
    "Healthy life expectancy": "life_expectancy",
    "Freedom to make life choices": "freedom",
    "Perceptions of corruption": "perception_corruption",
    "Generosity": "generosity"
}, inplace=True)

df_2018.rename(columns={
    "Country or region": "country_or_region",
    "Overall rank": "rank",
    "Score": "score",
    "Social support": "social_support",
    "GDP per capita": "gdp_per_capita",
    "Healthy life expectancy": "life_expectancy",
    "Freedom to make life choices": "freedom",
    "Perceptions of corruption": "perception_corruption",
    "Generosity": "generosity"
}, inplace=True)

df_2017.rename(columns={
    "Country": "country",
    "Happiness.Rank": "rank",
    "Happiness.Score": "score",
    "Economy..GDP.per.Capita.": "gdp_per_capita",
    "Family": "family",
    "Health..Life.Expectancy.": "life_expectancy",
    "Freedom": "freedom",
    "Trust..Government.Corruption.": "perception_corruption",
    "Generosity": "generosity",
    "Dystopia.Residual": "dystopia_residual"
}, inplace=True)

df_2016.rename(columns={
    "Country": "country",
    "Region": "region",
    "Happiness Rank": "rank",
    "Happiness Score": "score",
    "Economy (GDP per Capita)": "gdp_per_capita",
    "Family": "family",
    "Health (Life Expectancy)": "life_expectancy",
    "Freedom": "freedom",
    "Trust (Government Corruption)": "perception_corruption",
    "Generosity": "generosity",
    "Dystopia Residual": "dystopia_residual"
}, inplace=True)

df_2015.rename(columns={
    "Country": "country",
    "Region": "region",
    "Happiness Rank": "rank",
    "Happiness Score": "score",
    "Economy (GDP per Capita)": "gdp_per_capita",
    "Family": "family",
    "Health (Life Expectancy)": "life_expectancy",
    "Freedom": "freedom",
    "Trust (Government Corruption)": "perception_corruption",
    "Generosity": "generosity",
    "Dystopia Residual": "dystopia_residual"
}, inplace=True)

df = [df_2015, df_2016, df_2017, df_2018, df_2019]

known_countries = set(pd.concat([df_2015["country"], df_2016["country"], df_2017["country"]]).dropna().unique())
known_regions = set(pd.concat([df_2015["region"], df_2016["region"]]).dropna().unique())

def classify_by_reference(df):
    if "country_or_region" in df.columns:
        df["country"] = df["country_or_region"].apply(lambda x: x if x in known_countries else None)
        df["region"] = df["country_or_region"].apply(lambda x: x if x in known_regions else None)
    return df

df_2018 = classify_by_reference(df_2018)
df_2019 = classify_by_reference(df_2019)

df = pd.concat(df, ignore_index=True)

map_region = (df.dropna(subset=["region"])
                .drop_duplicates(subset=["country"], keep="first")
                .set_index("country")["region"]
                .to_dict())

df["region"] = df["region"].fillna(df["country"].map(map_region))

manual_regions = {
    "Taiwan Province of China": "Eastern Asia",
    "Hong Kong S.A.R., China": "Eastern Asia",
    "Trinidad & Tobago": "Latin America and Caribbean",
    "Northern Cyprus": "Middle East and Northern Africa",
    "North Macedonia": "Central and Eastern Europe",
    "Gambia": "Sub-Saharan Africa"
}
df["region"] = df["region"].fillna(df["country"].map(manual_regions))

# Impute missing numeric values
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
cols_with_nulls = ['family','perception_corruption','dystopia_residual','social_support']

# Linear Regression imputation
predictors = {
    'dystopia_residual':['score','gdp_per_capita','life_expectancy','freedom'],
    'social_support':['score','rank','gdp_per_capita','freedom','life_expectancy'],
    'perception_corruption':['score','generosity','freedom','gdp_per_capita'],
    'family':['score','gdp_per_capita','life_expectancy','freedom']
}

for target, feats in predictors.items():
    mask_missing = df[target].isna()
    if mask_missing.sum() == 0:
        continue
    train_data = df.dropna(subset=[target]+feats)
    if len(train_data) == 0:
        continue
    model = LinearRegression().fit(train_data[feats], train_data[target])
    df.loc[mask_missing, target] = model.predict(df.loc[mask_missing, feats])

features_model = ['gdp_per_capita','family','freedom','life_expectancy','social_support']
df = df.dropna(subset=features_model).reset_index(drop=True)
features = ['gdp_per_capita','family','freedom','life_expectancy','social_support','score','is_train']

def json_serializer(data):
    return json.dumps(data).encode("utf-8")

if __name__ == "__main__":
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_SERVER],
        value_serializer=json_serializer
    )

    for i, row in df.iterrows():
        data = {feat: row[feat] for feat in features}
        producer.send(TOPIC_NAME, value=data)
        print(f"{i+1} enviado (is_train={row['is_train']})")
        time.sleep(0.1)

    producer.flush()
    print("Todos los datos enviados exitosamente")