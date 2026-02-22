# ================================
# ENGINE FAILURE PREDICTOR - PRODUCTION TRAINING PIPELINE
# ================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import mlflow
import mlflow.sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

import xgboost as xgb
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ================================
# LOAD DATA FROM HUGGING FACE
# ================================

Xtrain = pd.read_csv("hf://datasets/nv185001/Realtime-Engine-Failure-Predictor/Xtrain.csv")
Xtest = pd.read_csv("hf://datasets/nv185001/Realtime-Engine-Failure-Predictor/Xtest.csv")
ytrain = pd.read_csv("hf://datasets/nv185001/Realtime-Engine-Failure-Predictor/ytrain.csv")
ytest = pd.read_csv("hf://datasets/nv185001/Realtime-Engine-Failure-Predictor/ytest.csv")

ytrain = ytrain.values.ravel()
ytest = ytest.values.ravel()

numeric_features = [
    'Engine_rpm',
    'Lub_oil_pressure',
    'Fuel_pressure',
    'Coolant_pressure',
    'lub_oil_temp',
    'Coolant_temp'
]

# ================================
# CLASS IMBALANCE HANDLING
# ================================

scale_pos_weight = (ytrain == 0).sum() / (ytrain == 1).sum()

# ================================
# PREPROCESSING PIPELINE
# ================================

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features)
)

xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss"
)

pipeline = make_pipeline(preprocessor, xgb_model)

# ================================
# HYPERPARAMETER TUNING
# ================================

param_grid = {
    "xgbclassifier__n_estimators": [100, 150],
    "xgbclassifier__max_depth": [3, 4, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__colsample_bytree": [0.6, 0.8],
    "xgbclassifier__subsample": [0.7, 0.8],
    "xgbclassifier__reg_lambda": [0.5, 1.0]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

# ================================
# MLFLOW EXPERIMENT TRACKING
# ================================

mlflow.set_experiment("Engine_Failure_XGBoost")

with mlflow.start_run():

    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(Xtest)
    y_proba = best_model.predict_proba(Xtest)[:, 1]

    # Metrics
    accuracy = accuracy_score(ytest, y_pred)
    precision = precision_score(ytest, y_pred)
    recall = recall_score(ytest, y_pred)
    f1 = f1_score(ytest, y_pred)
    roc_auc = roc_auc_score(ytest, y_proba)

    print(classification_report(ytest, y_pred))

    # Log parameters
    mlflow.log_params(grid_search.best_params_)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    # ================================
    # CONFUSION MATRIX
    # ================================

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

    # ================================
    # ROC CURVE
    # ================================

    fpr, tpr, _ = roc_curve(ytest, y_proba)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig("roc_curve.png")
    plt.close()

    mlflow.log_artifact("roc_curve.png")

    # ================================
    # FEATURE IMPORTANCE
    # ================================

    model = best_model.named_steps["xgbclassifier"]

    plt.figure()
    plt.bar(range(len(numeric_features)), model.feature_importances_)
    plt.xticks(range(len(numeric_features)), numeric_features, rotation=90)
    plt.title("Feature Importance")
    plt.savefig("feature_importance.png")
    plt.close()

    mlflow.log_artifact("feature_importance.png")

    # Save model locally
    joblib.dump(best_model, "best_engine_failure_model.joblib")

    mlflow.sklearn.log_model(best_model, "model")

# ================================
# UPLOAD MODEL TO HUGGING FACE
# ================================

repo_id = "nv185001/pred-model"
repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

api.upload_file(
    path_or_fileobj="best_engine_failure_model.joblib",
    path_in_repo="best_engine_failure_model.joblib",
    repo_id=repo_id,
    repo_type=repo_type
)

print("Model training and tracking complete.")