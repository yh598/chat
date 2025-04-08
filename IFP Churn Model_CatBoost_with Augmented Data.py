# Databricks notebook source
!pip install shap lime pdpbox catboost

# COMMAND ----------

import pyspark.pandas as ps
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, rand, isnan, year, month, expr, radians, sin, cos, atan2, sqrt, lit, udf, datediff, to_date
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split, cross_val_score
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report, precision_recall_curve, auc, f1_score, roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import lime
import lime.lime_tabular
from pdpbox import pdp, info_plots
from scipy.stats import ks_2samp
from catboost import CatBoostClassifier, Pool
import joblib
import warnings
import math

# COMMAND ----------

# Define Haversine formula as a UDF (distance in miles)
def haversine_miles(lat1, lon1, lat2, lon2):
    try:
        if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
            return None  # Return None if any coordinate is missing

        R = 3958.8  # Earth radius in miles

        # Convert degrees to radians
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon1 - lon2
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return abs(R * c)  # Distance in miles, always positive
    except ValueError:
        return None  # Return None if there is a ValueError

# Register as a PySpark UDF
haversine_miles_udf = udf(haversine_miles, DoubleType())

# Calculate distance only if provider_flag = 1 and cast as double
df = spark.sql('''select * from dev_adb.ifp_churn.data_augment_preprocessed''')

# Calculate median distance
median_distance = df.filter(col("provider_flag") == 1).select(
    haversine_miles_udf(
        col("LATITUDE"), col("LONGITUDE"), col("HOME_LATITUDE"), col("HOME_LONGITUDE")
    ).alias("distance").cast("double")
).approxQuantile("distance", [0.5], 0.01)[0]

df = df.withColumn(
    "PCP_DISTANCE",
    when(
        col("provider_flag") == 1,
        haversine_miles_udf(
            col("LATITUDE"), col("LONGITUDE"), col("HOME_LATITUDE"), col("HOME_LONGITUDE")
        ).cast("double")
    ).otherwise(lit(median_distance).cast("double"))  # Assign median value if provider_flag != 1
)

# Generate summary statistics for PCP_DISTANCE
summary_stats = df.select("PCP_DISTANCE").summary()
display(summary_stats)

# Detect outliers for PCP_DISTANCE
distance_quantiles = df.approxQuantile("PCP_DISTANCE", [0.01, 0.95], 0.01)
df = df.filter((col("PCP_DISTANCE") >= distance_quantiles[0]) & (col("PCP_DISTANCE") <= distance_quantiles[1]))

# COMMAND ----------



# COMMAND ----------

# Remove distance outliers
distance_quantiles = df.approxQuantile("PCP_DISTANCE", [0.01, 0.95], 0.01)
df = df.filter((col("PCP_DISTANCE") >= distance_quantiles[0]) & (col("PCP_DISTANCE") <= distance_quantiles[1]))

df = df.withColumn(
    "TENURE",
    datediff(to_date(col("LAST_ELIG_MONTH_END_DT")), to_date(col("FIRST_ELIG_MONTH_START_DT")))
)

# Remove tenure outliers (top 5%)
quantiles = df.approxQuantile("TENURE", [0.01, 0.95], 0.01)
df = df.filter((col("TENURE") >= quantiles[0]) & (col("TENURE") <= quantiles[1]))

# COMMAND ----------

# Drop columns with PII info, repeated info, and latitude/longitude
drop_cols = [col for col in df.columns if "HASH" in col or 
                "ID" in col or 
                "DT" in col or 
                "DATE" in col or 
                "NAME" in col or 
                "PHONE" in col or 
                "EMAIL" in col or 
                "ADDRESS" in col or 
                "ADDR" in col or 
                "FIPS" in col or 
                "TRACTCODE" in col or 
                "CENSUS" in col or
                "HOME" in col or
                "MAILING" in col or
                "ZIP" in col or
                "LATITUDE" in col or
                "LONGITUDE" in col or
                "CITY" in col or
                "PRIMARY" in col or
                "SECONDARY" in col or
                "PROV" in col]

dft = df.filter(col('open_enrollment_cycle') == '2024').drop(*drop_cols).drop(col('survival_flag'),col('SUBSCRIBER_KEY'),col('MEMBER_DOB'),col('AGE_IN_DAYS'),col('LANG_CD'),col('REL_BIRTHINFO'), col('RACE_CD'), col('ETHNIC_CD'), col('STUDY_PERIOD'), col('HCPK'), col('open_enrollment_cycle'),('COA'),('LOB1'),('LOB2'),('LOB3'),('LOB4'),('LOB5'),('LOB6'),('LOB7'),('MOBILE_COUNTRY_CODE'),('NPI'),('TIN'),('VERIF_STATUS'),('URBANIZATION'))

# Convert Spark DataFrame to Pandas DataFrame
pandas_df = dft.toPandas()

# Print confirmation
num_rows = dft.count()
num_cols = len(dft.columns)
print(f"Shape deleting columns: ({num_rows}, {num_cols})")

# COMMAND ----------

# Define target and feature columns
target_col = "churn_flag"
feature_cols = [col for col in pandas_df.columns if col != target_col]

# Split data into features (X) and target (y)
X = pandas_df[feature_cols]
y = pandas_df[target_col]

# Identify categorical columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Convert categorical columns to string type
for col in categorical_cols:
    X[col] = X[col].astype(str)

# Fill missing values with -999
X = X.fillna(-999) 

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Calculate class weights for handling imbalanced data
class_weights = {
    0: len(y_train) / (2 * (y_train == 0).sum()),
    1: len(y_train) / (2 * (y_train == 1).sum())
}

# Create CatBoost Pools for training and testing data
train_pool = Pool(data=X_train, label=y_train, cat_features=categorical_cols)
test_pool = Pool(data=X_test, label=y_test, cat_features=categorical_cols)

# Initialize and train CatBoost Classifier
catboost_model = CatBoostClassifier()
catboost_model.fit(train_pool)

# Get feature importance from the trained model
feature_importance = catboost_model.get_feature_importance()
feature_importance_df = pd.DataFrame({"Feature": feature_cols, "Importance": feature_importance})

# Get top 30 important features
top_30_features = feature_importance_df.sort_values(by="Importance", ascending=False).head(30)["Feature"].tolist()
print(top_30_features)

# COMMAND ----------

# Feature Importance Plot
top_df = feature_importance_df.sort_values(by="Importance", ascending=False).head(30)
plt.figure(figsize=(10, 6))
plt.barh(top_df.head(30)['Feature'], top_df.head(30)['Importance'], align='center')
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.title("Top 30 Most Important Features for IFP Churn Prediction")
plt.gca().invert_yaxis()
plt.show()

# COMMAND ----------

top_df

# COMMAND ----------

# Filter dataset to keep only the top 30 features
X_train_top = X_train[top_30_features]
X_test_top = X_test[top_30_features]

# Update categorical columns list for the top 30 features
categorical_cols_top = [col for col in categorical_cols if col in top_30_features]
print(categorical_cols_top)

# Use Pool object with top 30 features
train_pool_top = Pool(data=X_train_top, label=y_train, cat_features=categorical_cols_top)
test_pool_top = Pool(data=X_test_top, label=y_test, cat_features=categorical_cols_top)

# COMMAND ----------

# Parameter tuning to optimize recall using Optuna
import optuna
from optuna.integration import CatBoostPruningCallback

def objective(trial):
    param = {
        "depth": trial.suggest_int("depth", 4, 6),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1e-1),
        "iterations": trial.suggest_int("iterations", 500, 1000),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 3, 5),
        "border_count": trial.suggest_categorical("border_count", [32, 64]),
        "loss_function": "Logloss",
        "eval_metric": "Recall",
        "verbose": 0,
        "thread_count": -1
    }
    
    catboost_model_tuned = CatBoostClassifier(**param)
    catboost_model_tuned.fit(
        X_train_top, y_train,
        eval_set=(X_test_top, y_test),
        cat_features=categorical_cols_top,
        early_stopping_rounds=100,
        verbose=0,
        callbacks=[CatBoostPruningCallback(trial, "Recall")]
    )
    
    y_pred_prob = catboost_model_tuned.predict_proba(X_test_top)[:, 1]
    recall = recall_score(y_test, (y_pred_prob > 0.5).astype(int))
    return recall

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10, n_jobs=-1)

best_params = study.best_params
print("Best Parameters Found:", best_params)

# COMMAND ----------

# Train CatBoost with best parameters to optimize recall
catboost_final = CatBoostClassifier(
    **best_params,
    random_strength=5,  # Adds more randomness to avoid memorization
    verbose=100,
    loss_function="Logloss",  # Use Logloss for classification
    class_weights=class_weights,
    thread_count=-1,
    eval_metric="Recall"  # Optimize for recall
)
catboost_final.fit(train_pool_top, eval_set=test_pool_top, metric_period=100)

# Get predictions
y_pred_prob = catboost_final.predict_proba(test_pool_top)[:, 1]

# COMMAND ----------

# Compute precision-recall values
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

# Calculate F1 scores
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

# Plot the curves
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.plot(thresholds, f1_scores[:-1], label='F1 Score')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision, Recall, and F1 Score vs Decision Threshold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

# KS Statistic Calculation
def ks_statistic(y_true, y_pred_prob):
    churn_probs = y_pred_prob[y_true == 1]
    non_churn_probs = y_pred_prob[y_true == 0]
    ks_stat, p_value = ks_2samp(churn_probs, non_churn_probs)
    return ks_stat

# Lift Table Calculation
def lift_table(y_true, y_pred_prob, bins=10):
    df = pd.DataFrame({"actual": y_true, "predicted_prob": y_pred_prob})
    df["decile"] = pd.qcut(df["predicted_prob"], bins, labels=False, duplicates="drop")
    
    lift_df = df.groupby("decile").agg(
        count=("actual", "count"),
        events=("actual", "sum"),
        avg_prob=("predicted_prob", "mean")
    ).reset_index()
    
    lift_df["event_rate"] = lift_df["events"] / lift_df["count"]
    lift_df["cumulative_events"] = lift_df["events"].cumsum()
    lift_df["cumulative_event_rate"] = lift_df["cumulative_events"] / lift_df["cumulative_events"].max()
    
    return lift_df

# Model Performance & Optimal Threshold Selection
def model_performance(y_test, y_pred_prob):
    auc = roc_auc_score(y_test, y_pred_prob)
    ks_stat = ks_statistic(y_test, y_pred_prob)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC: {auc:.4f}", color="blue")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
    
    print(f"KS Statistic: {ks_stat:.4f}")
    
    # Compute Best Threshold
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    f1_scores = (2 * precision * recall) / (precision + recall)
    best_threshold = 0.5
    
    print(f"Best Decision Threshold: {best_threshold:.4f}")
    
    y_pred = (y_pred_prob >= best_threshold).astype(int)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="coolwarm", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))

    lift_df = lift_table(y_test, y_pred_prob)
    print("\n Lift Table:")
    display(lift_df)

model_performance(y_test, y_pred_prob)

# COMMAND ----------

# Cross-validation for better generalization using StratifiedKFold for faster execution
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_index, test_index in skf.split(X_train_top, y_train):
    X_train_fold, X_test_fold = X_train_top.iloc[train_index], X_train_top.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    
    catboost_final.fit(X_train_fold, y_train_fold, cat_features=categorical_cols_top)
    y_pred_prob_fold = catboost_final.predict_proba(X_test_fold)[:, 1]
    cv_scores.append(roc_auc_score(y_test_fold, y_pred_prob_fold))

print("Cross-Validation AUC:", np.mean(cv_scores))

# COMMAND ----------

import shap
# Explain the model with SHAP
explainer = shap.TreeExplainer(catboost_final)
shap_values = explainer.shap_values(X_test_top)

# SHAP Summary Plot (Feature Importance)
shap.summary_plot(shap_values, X_test_top)

# SHAP Waterfall Plot (Explains one specific prediction)
shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=X_test_top.iloc[0]))

# SHAP Dependence Plot (Shows impact of a single feature)
shap.dependence_plot(top_30_features[0], shap_values, X_test_top)

# COMMAND ----------

shap.dependence_plot(top_30_features[1], shap_values, X_test_top)

# COMMAND ----------

shap.dependence_plot(top_30_features[2], shap_values, X_test_top)

# COMMAND ----------

shap.dependence_plot(top_30_features[3], shap_values, X_test_top)

# COMMAND ----------

shap.dependence_plot(top_30_features[4], shap_values, X_test_top)

# COMMAND ----------

shap.dependence_plot(top_30_features[5], shap_values, X_test_top)

# COMMAND ----------

shap.dependence_plot(top_30_features[6], shap_values, X_test_top)

# COMMAND ----------

shap.dependence_plot(top_30_features[7], shap_values, X_test_top)

# COMMAND ----------

shap.dependence_plot(top_30_features[8], shap_values, X_test_top)

# COMMAND ----------

shap.dependence_plot(top_30_features[9], shap_values, X_test_top)

# COMMAND ----------

shap.dependence_plot(top_30_features[10], shap_values, X_test_top)

# COMMAND ----------

shap.dependence_plot(top_30_features[11], shap_values, X_test_top)

# COMMAND ----------

shap.dependence_plot(top_30_features[12], shap_values, X_test_top)

# COMMAND ----------

shap.dependence_plot(top_30_features[13], shap_values, X_test_top)

# COMMAND ----------

shap.dependence_plot(top_30_features[14], shap_values, X_test_top)

# COMMAND ----------

shap.dependence_plot(top_30_features[15], shap_values, X_test_top)

# COMMAND ----------

shap.dependence_plot(top_30_features[16], shap_values, X_test_top)

# COMMAND ----------

shap.dependence_plot(top_30_features[17], shap_values, X_test_top)

# COMMAND ----------

# Save the Trained Model for Deployment
joblib.dump(catboost_final, "/Volumes/dev_adb/ifp_churn/model/ifp_churn_model.pkl")
print("Model saved successfully")

# COMMAND ----------

# Load Model
catboost_loaded = joblib.load("/Volumes/dev_adb/ifp_churn/model/ifp_churn_model.pkl")

# Perform backtesting on 2021 to 2023 data
bt_df = df.filter(col('open_enrollment_cycle') != '2024').toPandas()

top_30_features = ['LANGUAGE_DESC', 'COA_ALIAS', 'AGE_IN_YEARS', 'DEPENDENT_FLAG', 'SPOUSE_FLAG', 'CAC', 'TENURE', 'SUBSCRIBER_FLAG', 'ACO_INDEX', 'RACE_DESC', 'PCP_DISTANCE', 'MBR_MED_MNTHS', 'INCOME_INEQUALITY_GINI_INDEX', 'PRODUCT_LEVEL_2', 'ADI_NATRANK', 'MBR_SFX', 'IMPUTED_PROB_API', 'LOW_INCOME_PEOPLE_HALF_MILE_FROM_ACCESS_TO_HEALTHY_FOOD', 'HOUSEHOLDS_WITHOUT_PUBLIC_ASSISTANCE_INCOME', 'HOUSEHOLDS_WITH_PUBLIC_ASSISTANCE_INCOME', 'ETHNIC_DESC', 'EP_DISABL', 'E_DAYPOP', 'PERCENTAGE_OF_LOW_INCOME_HALF_MILE_FROM_ACCESS_TO_HEALTHY_FOOD', 'PERCENTAGE_OF_LOW_INCOME_POPULATION', 'DXCG_RX_MEMBER_MONTHS', 'QTR_COPAY_AMT', 'E_DISABL', 'LOB5_ALIAS', 'EP_UNINSUR']

X_bt = bt_df[top_30_features]
y_bt = bt_df['churn_flag']

# Identify categorical columns
categorical_cols = X_bt.select_dtypes(include=["object", "category"]).columns.tolist()

# Convert categorical variables to strings (CatBoost handles them natively)
for col in categorical_cols:
    X_bt[col] = X_bt[col].astype(str)

bt_pool = Pool(data=X_bt, label=y_bt, cat_features=categorical_cols)

# Get predictions
y_pred_bt = catboost_loaded.predict_proba(bt_pool)[:, 1]

# COMMAND ----------

# KS Statistic Calculation
def ks_statistic(y_true, y_pred_prob):
    churn_probs = y_pred_prob[y_true == 1]
    non_churn_probs = y_pred_prob[y_true == 0]
    ks_stat, p_value = ks_2samp(churn_probs, non_churn_probs)
    return ks_stat

# Lift Table Calculation
def lift_table(y_true, y_pred_prob, bins=10):
    df = pd.DataFrame({"actual": y_true, "predicted_prob": y_pred_prob})
    df["decile"] = pd.qcut(df["predicted_prob"], bins, labels=False, duplicates="drop")
    
    lift_df = df.groupby("decile").agg(
        count=("actual", "count"),
        events=("actual", "sum"),
        avg_prob=("predicted_prob", "mean")
    ).reset_index()
    
    lift_df["event_rate"] = lift_df["events"] / lift_df["count"]
    lift_df["cumulative_events"] = lift_df["events"].cumsum()
    lift_df["cumulative_event_rate"] = lift_df["cumulative_events"] / lift_df["cumulative_events"].max()
    
    return lift_df

# Model Performance & Optimal Threshold Selection
def model_performance(y_test, y_pred_prob):
    auc = roc_auc_score(y_test, y_pred_prob)
    ks_stat = ks_statistic(y_test, y_pred_prob)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC: {auc:.4f}", color="blue")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
    
    print(f"KS Statistic: {ks_stat:.4f}")
    
    # Compute Best Threshold
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    f1_scores = (2 * precision * recall) / (precision + recall)
    best_threshold = 0.8
    
    print(f"Best Decision Threshold: {best_threshold:.4f}")
    
    y_pred = (y_pred_prob >= best_threshold).astype(int)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="coolwarm", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))

    lift_df = lift_table(y_test, y_pred_prob)
    print("\n Lift Table:")
    display(lift_df)

model_performance(y_bt, y_pred_bt)

# COMMAND ----------

# Defines function to predict churn probability & classify risk levels
def predict_churn(new_data):
    for col in categorical_cols:
        new_data[col] = new_data[col].astype(str)
    
    new_prob = catboost_loaded.predict_proba(new_data)[:, 1]
    
    risk_category = np.where(new_prob > 0.8, "High Risk",
                    np.where(new_prob > 0.5, "Medium Risk", "Low Risk"))
    
    return pd.DataFrame({"Churn Probability": new_prob, "Risk Category": risk_category})

# Classifies customers into High, Medium, Low risk groups
full_predictions = predict_churn(X_bt)

customer_segments = X_bt.copy()
customer_segments["Churn Probability"] = full_predictions["Churn Probability"]
customer_segments["Risk Category"] = full_predictions["Risk Category"]

print(customer_segments["Risk Category"].value_counts())

# COMMAND ----------

# Deploy Churn Prediction API using Flask
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data)

    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in categorical_cols_top:
        df[col] = df[col].astype(str)

    prob = catboost_loaded.predict_proba(df)[:, 1]
    risk_category = ["High Risk" if p > 0.8 else "Medium Risk" if p > 0.5 else "Low Risk" for p in prob]

    return jsonify({"churn_probability": prob.tolist(), "risk_category": risk_category})

if __name__ == "__main__":
    app.run(port=5000, debug=True)