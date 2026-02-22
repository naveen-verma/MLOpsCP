%%writefile cp_master/model_building/prep.py
# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/nv185001/Realtime-Engine-Failure-Predictor/engine_data.csv"
master_dataset = pd.read_csv(DATASET_PATH)

# 1. Missing Values Check
# Check total missing values per column
print("Missing Values Count:\n")
print(master_dataset.isnull().sum())

# 2. Column Name Cleaning
# This standardizes column names (removes spaces with underscores, replaces special chars)
master_dataset.columns = (
    master_dataset.columns
    .str.replace(" ", "_")
    .str.replace(r"[^\w]", "", regex=True)
)

print("Updated Column Names:\n")
print(master_dataset.columns)


# 3. Outlier Detection
print("Even though no missing data exists, there are suspicious extreme values")
print("Example Findings:")
print(" Coolant Temperature:")
print(" -Max = 195°C (unrealistic for engines)")
print(" -Normal range = 70–110°C")
print(" Oil Pressure:")
print(" -Some values extremely close to zero")


print("Dataset loaded successfully.")

# Define the target variable for the classification task
target = 'Engine_Condition'

# List of numerical features in the dataset
numeric_features = [
    'Engine_rpm',       # The number of revolutions per minute (RPM) of the engine, indicating engine speed. It is defined in Revolutions per Minute (RPM).
    'Lub_oil_pressure',  # The pressure of the lubricating oil in the engine, essential for reducing friction and wear. It is defined in bar or kilopascals (kPa)
    'Fuel_pressure',   # The pressure at which fuel is supplied to the engine, critical for proper combustion. It is defined in bar or kilopascals (kPa)
    'Coolant_pressure', # The pressure of the engine coolant, affecting engine temperature regulation. It is defined in bar or kilopascals (kPa).
    'lub_oil_temp',      # The temperature of the lubricating oil, which impacts viscosity and engine performance. It is defined in degrees Celsius (°C).
    'Coolant_temp'  # The temperature of the engine coolant, crucial for preventing overheating. It is defined in degrees Celsius (°C).
]


# Define predictor matrix (X) using selected numeric features
X = master_dataset[numeric_features]

# Define target variable
y = master_dataset[target]

# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="nv185001/Realtime-Engine-Failure-Predictor",
        repo_type="dataset",
    )
