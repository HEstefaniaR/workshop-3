# Workshop-3 ETL

**Made by**: Estefanía Hernández Rojas

---

This project integrates **Machine Learning** with **Data Streaming** to develop a complete predictive system for happiness scores across different countries and years.

The goal is to train a model capable of predicting the **Happiness Score** using relevant data features and to monitor its performance.

# ETL Pipeline

<p align="center">
  <img src="https://github.com/HEstefaniaR/workshop-3/blob/92d677fe1db567fdd2764b8422c7ff3e01d9aeec/diagrams/ETL%20Pipeline.png" alt="workshop 3 ETL Pipeline Diagram" width="751">
</p>

1. **Extract:** Reading the 5 original CSV files.
2. **Load:** The CSVs are saved with the addition of a flag to separate training data (70%) from test data (30%).
3. **Full Transform:**
   - Renaming columns
   - Adding a year column and merging the five datasets into one
   - Classification of country and region
   - Imputation of missing values (linear regression)
4. **EDA:** The transformed data allowed for an analysis to make decisions prior to model training.
5. **Model Training:** The model was trained using scikit-learn. PCA was applied to reduce data dimensionality, and three components were used for model training.

# Kafka Pipeline

<p align="center">
  <img src="https://github.com/HEstefaniaR/workshop-3/blob/92d677fe1db567fdd2764b8422c7ff3e01d9aeec/diagrams/Kafka%20Pipeline.png" alt="workshop 3 Kafka Pipeline Diagram" width="751">
</p>

- **Producer:**
  - Reads the processed CSVs
  - Executes the transformations
  - Selects the relevant features for the model
  - Sends each record to the happiness_features topic in Kafka
- **Consumer:**
  - Consumes records from the topic
  - Scales the data and applies PCA
  - Predicts the Happiness Score using the trained model (.pkl)
  - Inserts the predictions and features into the MySQL database
- **Visualizations:**
  - Dashboard to monitor model metrics in real time

# Project Structure

```
.
├── README.md
├── data
│   ├── processed
│   │   ├── 2015_with_flag.csv
│   │   ├── 2016_with_flag.csv
│   │   ├── 2017_with_flag.csv
│   │   ├── 2018_with_flag.csv
│   │   └── 2019_with_flag.csv
│   └── raw
│       ├── 2015.csv
│       ├── 2016.csv
│       ├── 2017.csv
│       ├── 2018.csv
│       └── 2019.csv
├── models
│   └── happiness_score_pca_model.pkl
├── notebooks
│   └── EDA and Model Training.ipynb
├── requirements.txt
├── scripts
│   ├── consumer.py
│   └── producer.py
├── start_kafka.sh
└── visualizations
    └── dashboard.py
```

- `notebooks/EDA and Model Training.ipynb`: contains the EDA used to perform the transformations later applied in the Kafka producer, as well as the model training using the five selected features. From this notebook, the processed data is created, including a flag for the data used in model training, respecting the 70/30 split.
- `data/`: contains the datasets used. The raw data includes the datasets used in the EDA, and the processed data additionally includes a flag indicating which data was used for model training. These processed data are used in the Kafka pipeline.
- `models/`: contains the trained model that will be used by the Kafka consumer.
- `scripts/`: contains the scripts for the Kafka consumer and producer.
- `start_kafka.sh`: script to start the local Kafka server.
- `visualizations/`: contains the dashboard to monitor the model’s prediction performance.
- `requirements.txt`: contains the Python environment requirements.

# **Technologies**

- Python 3.12
- Pandas, NumPy, Scikit-learn
- Apache Kafka
- MySQL (Data Warehouse)
- Jupyter Notebook for EDA and model training
- Visualizations: Plotly / Dash

# **Key Decisions**

## **EDA**

- Other columns were used to fill in missing country or region values in certain datasets.
- Columns such as `social support`, `perception_corruption`, and `family` had high null values and are useful as model features, so they were imputed using linear regression.
- Possible predictors considered were `gdp per capita`, `family`, `life expectancy`, `freedom`, and `social support`, as they had significant correlation with `score`.
- Multicollinearity was detected among several potential predictors. To confirm this, a multicollinearity test was performed, and all potential predictors showed VIF > 10, indicating high multicollinearity.

## **Model Training**

- Due to high multicollinearity, PCA was applied to reduce data dimensionality and avoid both overfitting and multicollinearity.
- Three components were used since they preserved 90% of the variance.
- The components were defined as follows, based on the variables with the highest contribution quality:
  - **PC1**: Social and Economic Prosperity
  - **PC2**: Freedom
  - **PC3**: Family

# **How to Run the Project**

1. **Clone the repository:**

```bash
git clone https://github.com/HEstefaniaR/workshop-2
cd workshop-2
```

2. **Create a virtual environment:**

```bash
python -m venv env
source env/bin/activate # Mac/Linux  
env\Scripts\activate    # Windows
```

3. **Install the requirements:**

```
pip install -r requirements.txt
```

4. Edit the variables in `scripts/consumer.py` to match your local MySQL credentials:

```python
DB_CONFIG = {
	"host": "localhost",
	"user": "USER",
	"password": "PASSWORD",
	"database": "happiness_score_dw"
}
```

5. Edit the variables in `start_kafka.sh` to match the paths for your repository and Kafka installation:

```bash
KAFKA_DIR='PATH'
PROJECT_DIR='PATH'
```

6. Use the script to start Kafka:

```bash
./start_kafka.sh
```

7. In other terminal start the dasboard of metrics:

   ```
   python visualizations/dashboard.py
   ```
   And open the dashboard running on: [http://127.0.0.1:8050/](http://127.0.0.1:8050/)
