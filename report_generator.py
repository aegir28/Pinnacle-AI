import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from ydata_profiling import ProfileReport
from pycaret.classification import setup as clf_setup, compare_models as clf_compare, pull as clf_pull, save_model as save_clf_model
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, pull as reg_pull, save_model as save_reg_model
from llm_utils import detect_target_column, detect_task_type

def detect_feature_columns(df, target_column):
    return [col for col in df.columns if col != target_column]

def is_classification_task(series):
    return series.nunique() < 15 or series.dtype == 'object' or series.dtype.name == 'category'

def preprocess_target_classes(df, target_column):
    value_counts = df[target_column].value_counts()
    valid_classes = value_counts[value_counts >= 2].index
    return df[df[target_column].isin(valid_classes)]

def generate_pinnacle_ai_report(df):
    os.makedirs("generated_reports", exist_ok=True)

    # Step 1: Detect target column and task type
    target_column = detect_target_column(df)
    task_type = detect_task_type(df)
    feature_columns = detect_feature_columns(df, target_column)

    df = preprocess_target_classes(df, target_column)

    # Step 2: Generate EDA report
    eda_report_path = os.path.join("generated_reports", "eda_report.html")
    profile = ProfileReport(df[feature_columns + [target_column]], title="Exploratory Data Analysis Report", explorative=True)
    profile.to_file(eda_report_path)

    # Step 3: AutoML using PyCaret
    if is_classification_task(df[target_column]):
        clf_setup(data=df, target=target_column, html=False, session_id=42, verbose=False)
        best_model = clf_compare()
        result_df = clf_pull()
        model_path = os.path.join("generated_reports", "best_model_classification")
        save_clf_model(best_model, model_path)
    else:
        reg_setup(data=df, target=target_column, html=False, session_id=42, verbose=False)
        best_model = reg_compare()
        result_df = reg_pull()
        model_path = os.path.join("generated_reports", "best_model_regression")
        save_reg_model(best_model, model_path)

    # Step 4: Plot heatmap of model comparison
    model_img_path = os.path.join("generated_reports", "model_comparison.png")
    plt.figure(figsize=(12, 6))
    sns.heatmap(result_df.select_dtypes(include='number'), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Model Comparison Heatmap")
    plt.tight_layout()
    plt.savefig(model_img_path)
    plt.close()

    return eda_report_path, model_img_path, target_column, task_type, model_path + '.pkl'
