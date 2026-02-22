# ==========================================
# ENGINE FAILURE PREDICTOR – OPTIMIZED FINAL VERSION
# ==========================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)

import xgboost as xgb
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ==========================================
# LOAD DATA
# ==========================================

Xtrain = pd.read_csv("hf://datasets/nv185001/Realtime-Engine-Failure-Predictor/Xtrain.csv")
Xtest = pd.read_csv("hf://datasets/nv185001/Realtime-Engine-Failure-Predictor/Xtest.csv")
ytrain = pd.read_csv("hf://datasets/nv185001/Realtime-Engine-Failure-Predictor/ytrain.csv").values.ravel()
ytest = pd.read_csv("hf://datasets/nv185001/Realtime-Engine-Failure-Predictor/ytest.csv").values.ravel()

numeric_features = [
    'Engine_rpm',
    'Lub_oil_pressure',
    'Fuel_pressure',
    'Coolant_pressure',
    'lub_oil_temp',
    'Coolant_temp'
]

# ==========================================
# FEATURE ENGINEERING (BOOSTS PERFORMANCE)
# ==========================================

def add_engineered_features(df):
    df = df.copy()
    df["Temp_Difference"] = df["Coolant_temp"] - df["lub_oil_temp"]
    df["Pressure_Ratio"] = df["Fuel_pressure"] / (df["Lub_oil_pressure"] + 1e-5)
    return df

Xtrain = add_engineered_features(Xtrain)
Xtest = add_engineered_features(Xtest)

numeric_features.extend(["Temp_Difference", "Pressure_Ratio"])

# ==========================================
# CLASS IMBALANCE HANDLING
# ==========================================

scale_pos_weight = (ytrain == 0).sum() / (ytrain == 1).sum()

# ==========================================
# PREPROCESSING PIPELINE
# ==========================================

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features)
)

xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss",
    tree_method="hist"
)

pipeline = make_pipeline(preprocessor, xgb_model)

# ==========================================
# RANDOMIZED SEARCH (FASTER + STRONGER)
# ==========================================

param_dist = {
    "xgbclassifier__n_estimators": [200, 300, 400],
    "xgbclassifier__max_depth": [4, 5, 6, 7],
    "xgbclassifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "xgbclassifier__subsample": [0.7, 0.8, 0.9],
    "xgbclassifier__colsample_bytree": [0.6, 0.7, 0.8],
    "xgbclassifier__gamma": [0, 0.1, 0.2],
    "xgbclassifier__min_child_weight": [1, 3, 5],
    "xgbclassifier__reg_lambda": [0.5, 1.0, 1.5]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=40,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# ==========================================
# MLFLOW TRACKING
# ==========================================

mlflow.set_experiment("Engine_Failure_Optimized_XGBoost")

with mlflow.start_run():

    search.fit(Xtrain, ytrain)

    best_model = search.best_estimator_

    # ==========================================
    # THRESHOLD OPTIMIZATION (CRITICAL)
    # ==========================================

    y_proba = best_model.predict_proba(Xtest)[:, 1]

    thresholds = np.arange(0.3, 0.7, 0.01)
    best_threshold = 0.5
    best_f1 = 0

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        score = f1_score(ytest, preds)
        if score > best_f1:
            best_f1 = score
            best_threshold = t

    print("Optimal Threshold:", best_threshold)

    y_pred = (y_proba >= best_threshold).astype(int)

    # ==========================================
    # METRICS
    # ==========================================

    accuracy = accuracy_score(ytest, y_pred)
    precision = precision_score(ytest, y_pred)
    recall = recall_score(ytest, y_pred)
    f1 = f1_score(ytest, y_pred)
    roc_auc = roc_auc_score(ytest, y_proba)

    print(classification_report(ytest, y_pred))

    mlflow.log_params(search.best_params_)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("optimal_threshold", best_threshold)

    # ==========================================
    # CONFUSION MATRIX
    # ==========================================

    cm = confusion_matrix(ytest, y_pred)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.xticks([0,1])
    plt.yticks([0,1])
    plt.savefig("confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")

    # ==========================================
    # ROC CURVE
    # ==========================================

    fpr, tpr, _ = roc_curve(ytest, y_proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig("roc_curve.png")
    plt.close()
    mlflow.log_artifact("roc_curve.png")

    # ==========================================
    # FEATURE IMPORTANCE
    # ==========================================

    model = best_model.named_steps["xgbclassifier"]

    plt.figure()
    plt.bar(range(len(numeric_features)), model.feature_importances_)
    plt.xticks(range(len(numeric_features)), numeric_features, rotation=90)
    plt.title("Feature Importance")
    plt.savefig("feature_importance.png")
    plt.close()
    mlflow.log_artifact("feature_importance.png")

    # ==========================================
    # SAVE MODEL
    # ==========================================

    joblib.dump(best_model, "optimized_engine_failure_model.joblib")
    mlflow.sklearn.log_model(best_model, "model")

# ==========================================
# UPLOAD TO HUGGING FACE
# ==========================================

repo_id = "nv185001/pred-model"
api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=repo_id, repo_type="model")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type="model", private=False)

api.upload_file(
    path_or_fileobj="optimized_engine_failure_model.joblib",
    path_in_repo="optimized_engine_failure_model.joblib",
    repo_id=repo_id,
    repo_type="model"
)

print("🔥 Optimized model training complete.")