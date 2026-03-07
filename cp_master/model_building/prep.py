# for data manipulation
import pandas as pd
import sklearn
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
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
data = master_dataset.copy()
#------------------------------------------------------------------------------
warnings.filterwarnings("ignore")
# Removes the limit for the number of displayed columns
pd.set_option("display.max_columns", None)
# set the limit for the number of displayed rows to 100
pd.set_option("display.max_rows", 100)

# Data Overview
# View the first and last 5 rows of the dataset
data.head()
data.tail()
# Understand the shape of the dataset
print(f"There are {data.shape[0]} rows and {data.shape[1]} columns.")

# Check the data types of the columns for the dataset
data.info()

 #Statistical summary of the data
# Let's check the statistical summary of the data.
data.describe(include="all").T

# check for the duplicate values in the dataset
duplicate_count = data.duplicated().sum()
print(f"There are {duplicate_count} duplicate rows in the dataset.")

#checking for the missing values in the dataset
missing_values_count = data.isnull().sum()
print("Missing Values Count:\n")
print(missing_values_count)

# 2. Column Name Cleaning
# This standardizes column names (removes spaces with underscores, replaces special chars)
data.columns = (
    data.columns
    .str.replace(" ", "_")
    .str.replace(r"[^\w]", "", regex=True)
)

print("Updated Column Names:\n")
print(data.columns)

# List of numerical features in the dataset
numeric_features = [
    'Engine_rpm',       # The number of revolutions per minute (RPM) of the engine, indicating engine speed. It is defined in Revolutions per Minute (RPM).
    'Lub_oil_pressure',  # The pressure of the lubricating oil in the engine, essential for reducing friction and wear. It is defined in bar or kilopascals (kPa)
    'Fuel_pressure',   # The pressure at which fuel is supplied to the engine, critical for proper combustion. It is defined in bar or kilopascals (kPa)
    'Coolant_pressure', # The pressure of the engine coolant, affecting engine temperature regulation. It is defined in bar or kilopascals (kPa).
    'lub_oil_temp',      # The temperature of the lubricating oil, which impacts viscosity and engine performance. It is defined in degrees Celsius (°C).
    'Coolant_temp'  # The temperature of the engine coolant, crucial for preventing overheating. It is defined in degrees Celsius (°C).
]

# Define the target variable for the classification task
target = 'Engine_Condition'

# Exploratory Data Analysis (EDA)
# Univariate Analysis
# function to plot a boxplot and a histogram along the same scale.

def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to the show density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a star will indicate the mean value of the column
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram

# Plotting the boxplot and histogram for each numeric feature
for feature in numeric_features:
    print(f"Analyzing feature: {feature}")
    histogram_boxplot(data, feature)
    plt.show()


# Bivariate ANalysis
# Correlation matrix
cols_list = data.select_dtypes(include=np.number).columns.tolist()

plt.figure(figsize=(10, 5))
sns.heatmap(
    data[cols_list].corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral"
)
plt.show()

# Let's check the distribution of our target variable 
# i.e Engine_Condition with the numeric columns
for feature in numeric_features:
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=data[feature], y=data[target])
    plt.title(f"Scatterplot of {feature} vs {target}")
    plt.show()

# Outlier Check
# Outlier detection using boxplots for each numeric feature
numeric_columns = data.select_dtypes(include=np.number).columns.tolist()

for i, variable in enumerate(numeric_columns):
    plt.subplot(4, 4, i + 1)
    plt.boxplot(data[variable], whis=1.5)
    plt.tight_layout()
    plt.title(variable)

plt.show()
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Automated EDA using ydata-profiling (for MLOps pipeline)

print("Generating automated EDA report...")

# Create reports directory
os.makedirs("reports", exist_ok=True)

# Generate profiling report
profile = ProfileReport(
    data,
    title="Engine Failure Predictor - Automated EDA",
    explorative=True,  # enables deeper analysis
    correlations={
        "pearson": {"calculate": True},
        "spearman": {"calculate": True},
        "phi_k": {"calculate": True},
        "cramers": {"calculate": True},
    },
    missing_diagrams={
        "matrix": True,
        "bar": True,
        "heatmap": True
    }
)

# Save report
EDA_OUTPUT = "eda_report.html"
profile.to_file(EDA_OUTPUT)

print(f"EDA report generated: {EDA_OUTPUT}")
#------------------------------------------------------------------------------






# Define predictor matrix (X) using selected numeric features
X = data[numeric_features]

# Define target variable
y = data[target]

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
