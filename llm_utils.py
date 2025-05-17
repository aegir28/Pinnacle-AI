import pandas as pd

def detect_target_column(df):
    """
    Automatically detect the target column based on heuristic:
    - Choose the column with lowest number of unique values (excluding index-like or ID-like columns).
    """
    non_id_columns = [col for col in df.columns if not any(keyword in col.lower() for keyword in ['id', 'index'])]
    target_col = None
    min_unique = float('inf')

    for col in non_id_columns:
        unique_vals = df[col].nunique()
        if unique_vals < min_unique and unique_vals > 1:  # Avoid constant columns
            min_unique = unique_vals
            target_col = col

    return target_col

def detect_task_type(df):
    """
    Determine if the task is classification or regression based on target column:
    - Classification: if target has less than 15 unique values or is categorical.
    - Regression: otherwise.
    """
    target_col = detect_target_column(df)
    if target_col is None:
        return "unknown"

    target_series = df[target_col]

    if target_series.nunique() < 15 or target_series.dtype == 'object' or pd.api.types.is_categorical_dtype(target_series):
        return "classification"
    else:
        return "regression"
