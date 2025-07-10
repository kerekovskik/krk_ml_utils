import os
import pandas as pd
from .preprocessing import MinMax, ZScore, RobustScaler, CyclicScaler
import numpy as np
from scipy.stats import shapiro
import json
import matplotlib.pyplot as plt
from typing import Optional, List


def analyze_string_columns(df):
    """
    Analyzes string columns in a pandas DataFrame and prints the number of unique values per column.

    Args:
        df (pd.DataFrame): The input DataFrame.
    """

    #
    string_columns = df.select_dtypes(include=["object", "string"]).columns

    if string_columns.empty:
        print("No string columns found in the DataFrame.")
        return

    for col in string_columns:
        num_unique = df[col].nunique()
        print(f"Column '{col}': {num_unique} unique values")


def auto_dummy_encode(df):
    # Identify string columns
    string_columns = df.select_dtypes(include=["object"]).columns

    # One-hot encode string columns
    df_encoded = pd.get_dummies(df, columns=string_columns, drop_first=True)
    return df_encoded


def label_encode(df, column_list, save_dir):
    """
    Label encodes the specified columns in a pandas DataFrame. For each column in `column_list`, it replaces the unique values with integers starting from 0.
    If a column is not present in the DataFrame, it will be skipped.
    For each encoded column, it saves a mapping forward and backward transformation in a JSON file in the specified `save_dir`.
    The encoded columns will be moved to the right of the DataFrame, maintaining the original order of other columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_list (list): List of column names to label encode.

    Returns:
        pd.DataFrame: DataFrame with specified columns label encoded.
    """

    # Filter the columns to only those that exist in the DataFrame
    existing_cols = [col for col in column_list if col in df.columns]

    # Build a dictionary saving forward and backward mapping for each column
    label_info = {}
    for col in existing_cols:
        unique_vals = df[col].dropna().unique()
        mapping = {str(val): idx for idx, val in enumerate(unique_vals)}
        reverse_mapping = {idx: str(val) for val, idx in mapping.items()}
        label_info[col] = {"forward": mapping, "backward": reverse_mapping}

        # Apply label encoding
        df[col] = df[col].astype(str)  # Ensure all values are strings
        # Map the values to integers using the forward mapping
        df[col] = df[col].map(mapping).astype("Int64")  # Use 'Int64' to handle NaNs

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save the label_info as a JSON file
    json_path = os.path.join(save_dir, "label_encoded_columns.json")
    with open(json_path, "w") as file:
        json.dump(label_info, file)

    # Move the encoded columns to the right of the DataFrame
    encoded_cols = [col for col in df.columns if col in existing_cols]
    other_cols = [col for col in df.columns if col not in existing_cols]
    df = df[other_cols + encoded_cols]
    print(f"Label encoded columns: {existing_cols} and saved mappings to {json_path}")

    return df


def undo_label_encode(df, save_dir: str):
    """
    Reverses the label encoding transformation performed by label_encode.

    Reads the JSON file saved by label_encode to determine the forward and backward mappings
    for each label encoded column, then reconstructs the original categorical column.
    """
    json_path = os.path.join(save_dir, "label_encoded_columns.json")

    # Return the input dataframe if the JSON file doesn't exist
    if not os.path.exists(json_path):
        return df

    with open(json_path, "r") as file:
        label_info = json.load(file)

    for col, info in label_info.items():
        if col in df.columns:
            # Reverse mapping to restore original values
            reverse_mapping = info["backward"]
            # Map to a lambda that converts the integer label to a string before applying
            # the reverse mapping
            df[col] = df[col].map(
                lambda x: reverse_mapping.get(str(int(x)) if pd.notna(x) else None, x)
            )

    return df


def redo_label_encode(df, save_dir: str):
    """
    Re-applies the label encoding transformation using saved mappings.

    Reads the JSON file saved by label_encode to determine the forward mappings
    for each label encoded column, then applies the same transformation to a copy of the dataframe.

    Args:
        df (pd.DataFrame): The input DataFrame.
        save_dir (str): Directory where the mapping information is saved.

    Returns:
        pd.DataFrame: A copy of the DataFrame with label encoding reapplied.
    """
    df_copy = df.copy()
    json_path = os.path.join(save_dir, "label_encoded_columns.json")

    # Return the copy if the JSON file doesn't exist
    if not os.path.exists(json_path):
        return df_copy

    with open(json_path, "r") as file:
        label_info = json.load(file)

    for col, info in label_info.items():
        if col in df_copy.columns:
            # Apply forward mapping to re-encode the values
            forward_mapping = info["forward"]
            df_copy[col] = df_copy[col].astype(str)  # Ensure all values are strings
            df_copy[col] = df_copy[col].map(forward_mapping).astype("Int64")

    return df_copy


def dummy_encode_and_save(df, column_list, save_dir):
    # Filter the columns to only those that exist in the DataFrame
    existing_cols = [col for col in column_list if col in df.columns]

    # Build a dictionary saving base category info for each column
    dummy_info = {}
    for col in existing_cols:
        # Get sorted unique non-null values as strings
        vals = sorted(df[col].dropna().astype(str).unique())
        if vals:
            base_val = vals[0]
        else:
            base_val = None
        dummy_info[col] = {"base": base_val}

    # Dummy encode the specified columns
    df_encoded = pd.get_dummies(df, columns=existing_cols, drop_first=True)

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save the dummy_info as a JSON file
    json_path = os.path.join(save_dir, "dummy_encoded_columns.json")
    with open(json_path, "w") as file:
        json.dump(dummy_info, file)

    return df_encoded


def undo_dummy_encode(df, save_dir: str):
    """
    Reverses the dummy encoding transformation performed by dummy_encode_and_save.

    Reads the JSON file saved by dummy_encode_and_save to determine the base category
    for each dummy encoded column, then reconstructs the original categorical column.
    """
    json_path = os.path.join(save_dir, "dummy_encoded_columns.json")

    # Return the input dataframe if the JSON file doesn't exist
    if not os.path.exists(json_path):
        return df

    with open(json_path, "r") as file:
        dummy_info = json.load(file)

    for col, info in dummy_info.items():
        # Find all dummy columns corresponding to the original column 'col'
        dummy_columns = [c for c in df.columns if c.startswith(f"{col}_")]
        if not dummy_columns:
            continue

        def restore_category(row):
            # If all dummy columns are 0, return the recorded base category
            if row[dummy_columns].sum() == 0:
                return info["base"]
            else:
                for dummy in dummy_columns:
                    if row[dummy] == 1:
                        return dummy.split(f"{col}_", 1)[-1]
                return np.nan

        df[col] = df.apply(restore_category, axis=1)
        df.drop(columns=dummy_columns, inplace=True)

    return df


def reDummyEncodeData(df, save_dir: str):
    """
    Re-applies the dummy encoding transformation using saved base category information.

    Reads the JSON file saved by dummy_encode_and_save to determine the base category
    for each original column, then applies dummy encoding to a copy of the dataframe.

    Args:
        df (pd.DataFrame): The input DataFrame.
        save_dir (str): Directory where the base category information is saved.

    Returns:
        pd.DataFrame: A copy of the DataFrame with dummy encoding reapplied.
    """
    df_copy = df.copy()
    json_path = os.path.join(save_dir, "dummy_encoded_columns.json")

    # Return the copy if the JSON file doesn't exist
    if not os.path.exists(json_path):
        return df_copy

    with open(json_path, "r") as file:
        dummy_info = json.load(file)

    # Get the columns that need to be dummy encoded
    columns_to_encode = [col for col in dummy_info.keys() if col in df_copy.columns]

    if columns_to_encode:
        # Apply dummy encoding with drop_first=True to match the original transformation
        df_copy = pd.get_dummies(df_copy, columns=columns_to_encode, drop_first=True)

    return df_copy


def minMaxData(df, column_list: list, save_dir: str):
    """
    Min-max normalizes the specified columns in a pandas DataFrame.
    """

    os.makedirs(save_dir, exist_ok=True)

    for col in column_list:
        scaler = MinMax()
        scaler.fit(df[col])
        df[col] = scaler.normalize(df[col])
        scaler.save(f"{save_dir}/{col}_minmax.npz")

    return df


def unMinMaxData(df, save_dir: str):

    for col in df.columns:
        if os.path.exists(f"{save_dir}/{col}_minmax.npz"):
            scaler = MinMax()
            scaler.load(f"{save_dir}/{col}_minmax.npz")
            df[col] = scaler.unnormalize(df[col])
        else:
            print(f"MinMax file for {col} not found")
    return df


def reMinMaxData(df, save_dir: str):
    """
    Re-applies min-max normalization using saved scaler parameters.

    Loads the saved MinMax scaler parameters and applies the same transformation
    to a copy of the dataframe.

    Args:
        df (pd.DataFrame): The input DataFrame.
        save_dir (str): Directory where the scaler parameters are saved.

    Returns:
        pd.DataFrame: A copy of the DataFrame with min-max normalization reapplied.
    """
    df_copy = df.copy()

    for col in df_copy.columns:
        scaler_path = f"{save_dir}/{col}_minmax.npz"
        if os.path.exists(scaler_path):
            scaler = MinMax()
            scaler.load(scaler_path)
            df_copy[col] = scaler.normalize(df_copy[col])

    return df_copy


def zscoreData(df, column_list: list, save_dir: str):
    """
    Z-score normalizes the specified columns in a pandas DataFrame.
    """
    os.makedirs(save_dir, exist_ok=True)

    for col in column_list:
        scaler = ZScore()
        scaler.fit(df[col])
        df[col] = scaler.normalize(df[col])
        scaler.save(f"{save_dir}/{col}_zscore.npz")

    return df


def unZscoreData(df, save_dir: str):
    """
    Un-normalizes the specified columns in a pandas DataFrame using Z-score normalization.
    """
    for col in df.columns:
        if os.path.exists(f"{save_dir}/{col}_zscore.npz"):
            scaler = ZScore()
            scaler.load(f"{save_dir}/{col}_zscore.npz")
            df[col] = scaler.unnormalize(df[col])
        else:
            print(f"ZScore file for {col} not found")
    return df


def reZscoreData(df, save_dir: str):
    """
    Re-applies z-score normalization using saved scaler parameters.

    Loads the saved ZScore scaler parameters and applies the same transformation
    to a copy of the dataframe.

    Args:
        df (pd.DataFrame): The input DataFrame.
        save_dir (str): Directory where the scaler parameters are saved.

    Returns:
        pd.DataFrame: A copy of the DataFrame with z-score normalization reapplied.
    """
    df_copy = df.copy()

    for col in df_copy.columns:
        scaler_path = f"{save_dir}/{col}_zscore.npz"
        if os.path.exists(scaler_path):
            scaler = ZScore()
            scaler.load(scaler_path)
            df_copy[col] = scaler.normalize(df_copy[col])

    return df_copy


def robustData(df, column_list: list, save_dir: str):
    """
    Robust scales the specified columns in a pandas DataFrame.
    """

    os.makedirs(save_dir, exist_ok=True)

    for col in column_list:
        scaler = RobustScaler()
        scaler.fit(df[col])
        df[col] = scaler.normalize(df[col])
        scaler.save(f"{save_dir}/{col}_robust.npz")
    return df


def unRobustData(df, save_dir: str):
    """
    Un-scales the specified columns in a pandas DataFrame using robust scaling.
    """
    for col in df.columns:
        if os.path.exists(f"{save_dir}/{col}_robust.npz"):
            scaler = RobustScaler()
            scaler.load(f"{save_dir}/{col}_robust.npz")
            df[col] = scaler.unnormalize(df[col])
        else:
            print(f"Robust file for {col} not found")
    return df


def reRobustData(df, save_dir: str):
    """
    Re-applies robust scaling using saved scaler parameters.

    Loads the saved RobustScaler parameters and applies the same transformation
    to a copy of the dataframe.

    Args:
        df (pd.DataFrame): The input DataFrame.
        save_dir (str): Directory where the scaler parameters are saved.

    Returns:
        pd.DataFrame: A copy of the DataFrame with robust scaling reapplied.
    """
    df_copy = df.copy()

    for col in df_copy.columns:
        scaler_path = f"{save_dir}/{col}_robust.npz"
        if os.path.exists(scaler_path):
            scaler = RobustScaler()
            scaler.load(scaler_path)
            df_copy[col] = scaler.normalize(df_copy[col])

    return df_copy


def describeDataframe(df):
    """
    Describes a pandas DataFrame with additional details.

    Args:
        df (pd.DataFrame): The input DataFrame.
    """

    print("DataFrame Summary:")
    for col in df.columns:
        dtype = df[col].dtype
        unique_count = df[col].nunique()

        print_str = ""

        print_str += (
            f"Column: {col} | Data Type: {dtype} | Unique Values: {unique_count}"
        )
        # print(f"Column: {col}")
        # print(f"  Data Type: {dtype}")
        # print(f"  Unique Values: {unique_count}")

        if pd.api.types.is_numeric_dtype(df[col]):
            mode_value = "N/A"
            mode_series = df[col].mode()
            if not mode_series.empty:
                mode_value = mode_series.iloc[0]
            # print(f"  Mode ): {mode_value}")

            mean_value = df[col].mean()
            min_value = df[col].min()
            max_value = df[col].max()
            std_dev = df[col].std()
            print_str += f" | Mode: {mode_value} | Mean: {mean_value:.2f} | Min: {min_value} | Max: {max_value} | Std. Dev.: {std_dev:.2f}"
        elif df[col].dtype == object:
            # Calculate string lengths only for non-null values
            lengths = df[col].dropna().astype(str).str.len()
            if not lengths.empty:
                avg_length = lengths.mean()
                min_length = lengths.min()
                max_length = lengths.max()
            else:
                avg_length = min_length = max_length = 0

            print_str += f" | Avg. String Length: {avg_length:.2f} | Min String Length: {min_length} | Max String Length: {max_length}"

            # print(f"  Average String Length: {avg_length}")
            # print(f"  Min String Length: {min_length}")
            # print(f"  Max String Length: {max_length}")

        print(print_str)


def identify_categorical_columns(
    df,
    unique_threshold_string=10,
    unique_threshold_numeric=20,
    unique_percentage_threshold=0.05,
    user_categorical_cols=None,
):
    """
    Identifies potentially categorical columns in a Pandas DataFrame based on various rules.

    Args:
        df (pd.DataFrame): The input DataFrame.
        unique_threshold_string (int): Max unique values for string columns to be flagged (default 10).
        unique_threshold_numeric (int): Max unique values for numeric columns to be flagged (default 20).
        unique_percentage_threshold (float): Max percentage of unique values relative to total rows (default 0.05 or 5%).
        user_categorical_cols (list or None): List of column names to *always* treat as categorical.

    Returns:
        list: A list of column names that were flagged as potentially categorical.
        Prints messages to the console indicating flagged columns and reasons.
    """
    flagged_columns = []

    if user_categorical_cols is not None:
        for col in user_categorical_cols:
            if col in df.columns:
                print(
                    f"User-defined Categorical Column: '{col}' (as specified by user)"
                )
                flagged_columns.append(col)
            else:
                print(
                    f"Warning: User specified column '{col}' as categorical, but it's not in the DataFrame."
                )

    for col in df.columns:
        if col in flagged_columns:  # Skip if already flagged by user
            continue

        dtype = df[col].dtype

        if pd.api.types.is_categorical_dtype(dtype):  # Rule 4: 'category' dtype
            print(f"Categorical Column: '{col}' (dtype is 'category')")
            flagged_columns.append(col)
        elif pd.api.types.is_bool_dtype(dtype):  # Rule 3: Boolean dtype
            print(
                f"Boolean Column (Categorical): '{col}' (dtype is 'bool') - Likely no further encoding needed."
            )
            flagged_columns.append(col)
        elif (
            dtype == "object"
        ):  # Rule 1: String/Object with low unique count (using 'object' to catch strings effectively in pandas context)
            unique_count = df[col].nunique()
            if unique_count <= unique_threshold_string:
                print(
                    f"Potentially Categorical Column: '{col}' (string/object type, {unique_count} unique values <= {unique_threshold_string})"
                )
                flagged_columns.append(col)
        elif pd.api.types.is_numeric_dtype(dtype):
            unique_count = df[col].nunique()
            if (unique_count <= unique_threshold_numeric) or (
                (unique_count / len(df)) <= unique_percentage_threshold
                and unique_count <= 50
            ):  # Revised numeric rule
                if unique_count <= unique_threshold_numeric:
                    print(
                        f"Potentially Categorical Column: '{col}' (numeric type, {unique_count} unique values <= {unique_threshold_numeric})"
                    )
                else:  # Implies percentage condition was met
                    print(
                        f"Potentially Categorical Column: '{col}' (numeric type, {unique_count} unique values, {unique_count/len(df):.2%} unique percentage <= {unique_percentage_threshold:.2%}, and unique_count <= 50)"
                    )
                flagged_columns.append(col)

    # Print out the flagged columns to look like a list initialization in python
    print(f"\n\ndummy_list = ['" + "', '".join(flagged_columns) + "']")
    return flagged_columns


def detect_outliers(data: pd.Series, iqr_factor: float = 1.5) -> float:
    """
    Detects the percentage of outliers in the data using the IQR method.

    Args:
        data (pd.Series): A numeric pandas Series.
        iqr_factor (float): Multiplier for the IQR to define outlier boundaries.

    Returns:
        float: Percentage of outliers in the data.
    """
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - iqr_factor * iqr
    upper_bound = q3 + iqr_factor * iqr
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return len(outliers) / len(data) * 100


def check_normality(
    data: pd.Series, alpha: float = 0.05, sample_limit: int = 5000
) -> bool:
    """
    Checks if data is normally distributed using the Shapiro-Wilk test.

    Args:
        data (pd.Series): A numeric pandas Series.
        alpha (float): Significance level for the normality test.
        sample_limit (int): Maximum number of samples to test (Shapiro's limit is 5000).

    Returns:
        bool: True if the data appears normally distributed, False otherwise.
    """
    if len(data) > sample_limit:
        sample = data.sample(sample_limit, random_state=42)
    else:
        sample = data

    try:
        _, p_value = shapiro(sample)
    except ValueError:
        # Shouldn't really happen since we've filtered out constant values earlier.
        return False

    return p_value > alpha


def recommend_scaling_methods(
    df: pd.DataFrame,
    outlier_threshold: float = 5.0,
    alpha: float = 0.05,
    sample_limit: int = 5000,
) -> dict:
    """
    Analyzes numerical DataFrame columns and recommends scaling methods.

    Evaluates each numerical column based on its distribution characteristics
    to determine if scaling is recommended and which method (z-score, robust,
    or min-max) would be most appropriate. Returns a dictionary of recommendations.

    Decision logic:
    1. Constant or binary columns: No scaling needed.
    2. Columns with >outlier_threshold% outliers: Robust scaling.
    3. Normally distributed columns: Z-score scaling.
    4. Other columns: Min-max scaling.

    Args:
        df (pd.DataFrame): Input DataFrame containing numerical columns.
        outlier_threshold (float): Percentage threshold to decide if robust scaling is needed.
        alpha (float): Significance level for the Shapiro-Wilk test.
        sample_limit (int): Maximum number of samples for the normality test.

    Returns:
        dict: A dictionary where each key is a column name and the value is another
              dictionary with keys "should_scale" (bool) and "recommended_method" (str).
    """
    recommendations = {}

    # Iterate through all numeric columns in the DataFrame
    for col in df.select_dtypes(include=np.number).columns:
        data = df[col].dropna()
        unique_vals = data.nunique()
        col_recommendation = {}

        # Check for constant or binary columns
        if unique_vals <= 1:
            col_recommendation["should_scale"] = False
            col_recommendation["recommended_method"] = (
                "No scaling needed (constant value)"
            )
            recommendations[col] = col_recommendation
            continue

        if unique_vals == 2:
            col_recommendation["should_scale"] = False
            col_recommendation["recommended_method"] = (
                "No scaling needed (binary column)"
            )
            recommendations[col] = col_recommendation
            continue

        # Outlier detection
        outlier_pct = detect_outliers(data)
        if outlier_pct > outlier_threshold:
            col_recommendation["should_scale"] = True
            col_recommendation["recommended_method"] = (
                f"robust scaling (> {outlier_threshold}% outliers detected)"
            )
            recommendations[col] = col_recommendation
            continue

        # Normality test using Shapiro-Wilk
        is_normal = check_normality(data, alpha, sample_limit)
        if is_normal:
            col_recommendation["should_scale"] = True
            col_recommendation["recommended_method"] = (
                "z-score scaling (normally distributed)"
            )
        else:
            col_recommendation["should_scale"] = True
            col_recommendation["recommended_method"] = (
                "min-max scaling (non-normal distribution without significant outliers)"
            )

        recommendations[col] = col_recommendation

    return recommendations


def print_recommendations(recommendations: dict) -> None:
    """
    Prints the scaling recommendations to standard output.

    Args:
        recommendations (dict): A dictionary with columns as keys and recommendation
                                information as values.
    """
    for col, rec in recommendations.items():
        print(f"Column: {col}")
        print(f"- Should be scaled: {'Yes' if rec['should_scale'] else 'No'}")
        print(f"- Recommended method: {rec['recommended_method']}")
        print("--------------------")

    # Build list initializations for robust_list, zscore_list, and minmax_list
    robust_list = [
        col
        for col, rec in recommendations.items()
        if "robust" in rec["recommended_method"].lower()
    ]
    zscore_list = [
        col
        for col, rec in recommendations.items()
        if "z-score" in rec["recommended_method"].lower()
    ]
    minmax_list = [
        col
        for col, rec in recommendations.items()
        if "min-max" in rec["recommended_method"].lower()
    ]

    print(f"\n\nrobust_list = {robust_list if len(robust_list) > 0 else None}")
    print(f"zscore_list = {zscore_list if len(zscore_list) > 0 else None}")
    print(f"minmax_list = {minmax_list if len(minmax_list) > 0 else None}")


def normalize_dataframe(
    df: pd.DataFrame,
    save_dir: str,
    zscore_list: list = None,
    robust_list: list = None,
    minmax_list: list = None,
    dummy_list: list = None,
    label_list: list = None,
):
    """
    Normalize dataframe columns and save the normalization parameters.
    """

    if zscore_list:
        df = zscoreData(df, zscore_list, save_dir)

    if robust_list:
        df = robustData(df, robust_list, save_dir)

    if minmax_list:
        df = minMaxData(df, minmax_list, save_dir)

    if dummy_list:
        df = dummy_encode_and_save(df, dummy_list, save_dir)

    if label_list:
        df = label_encode(df, label_list, save_dir)

    return df


def denorm_dataframe(df: pd.DataFrame, save_dir: str) -> pd.DataFrame:
    """
    Denormalize dataframe columns using the saved normalization parameters.
    """
    df = df.copy()
    df = unMinMaxData(df, save_dir)
    df = unZscoreData(df, save_dir)
    df = unRobustData(df, save_dir)
    df = undo_dummy_encode(df, save_dir)
    df = undo_label_encode(df, save_dir)
    df = undoCycleScale(df, save_dir)
    
    df = undo_feature_engineer(df, save_dir)
    df = undo_find_and_replace(df, save_dir)

    # Ensure the DataFrame is returned in its original order
    with open(os.path.join(save_dir, "column_mapping.json"), "r") as f:
        original_columns_mapping = json.load(f)

    original_columns = sorted(
        original_columns_mapping, key=lambda x: original_columns_mapping[x]
    )
    df = df[original_columns] if set(original_columns).issubset(df.columns) else df

    return df


class FindReplaceConfig:
    """
    Configuration class for find and replace operations.
    This class is used to store the configuration for find and replace operations
    in a structured way, allowing for easy serialization and deserialization.
    """

    def __init__(self, find_value: str, replace_value: str, column_list: list):
        self.find_value = find_value
        self.replace_value = replace_value
        self.column_list = column_list

    def to_dict(self):
        return {
            "find_value": self.find_value,
            "replace_value": self.replace_value,
            "column_list": self.column_list,
        }
  
      
class CyclicScaleConfig:
    """

    """
    def __init__(self, column_name, period):
        self.column_name = column_name
        self.period = period
        

def cycleScaleData(df: pd.DataFrame, cycle_list: Optional[List[CyclicScaleConfig]], save_dir: str) -> pd.DataFrame:
    """
    Applies cyclic scaling to specified columns using CyclicScaler.
    
    Creates sine and cosine components for each cyclic column, adds them as new columns
    with suffixes '_sine' and '_cosine', and removes the original column.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        cycle_list (Optional[List[CyclicScaleConfig]]): List of CyclicScaleConfig objects
                                                       specifying columns and their periods.
        save_dir (str): Directory to save the cyclic scaling parameters.
        
    Returns:
        pd.DataFrame: DataFrame with cyclic columns transformed to sine/cosine components.
    """
    if cycle_list is None:
        return df
        
    df = df.copy()
    os.makedirs(save_dir, exist_ok=True)
    
    for config in cycle_list:
        col_name = config.column_name
        period = config.period
        
        if col_name not in df.columns:
            print(f"Warning: Column '{col_name}' not found in DataFrame. Skipping.")
            continue
            
        # Create and fit the cyclic scaler
        scaler = CyclicScaler()
        scaler.fit(df[col_name], period)
        
        # Transform the data to sine/cosine components
        sin_cos_data = scaler.normalize(df[col_name])
        
        # Add new columns for sine and cosine components
        df[f"{col_name}_sine"] = sin_cos_data[:, 0]
        df[f"{col_name}_cosine"] = sin_cos_data[:, 1]
        
        # Remove the original column
        df.drop(columns=[col_name], inplace=True)
        
        # Save the scaler parameters
        scaler.save(f"{save_dir}/{col_name}_cyclic.npz")
        
        print(f"Applied cyclic scaling to column '{col_name}' with period {period}")
    
    return df


def undoCycleScale(df: pd.DataFrame, save_dir: str) -> pd.DataFrame:
    """
    Undoes cyclic scaling by reconstructing original columns from sine/cosine components.
    
    Looks for _cyclic.npz files in the save directory and reconstructs the original
    cyclic columns, then removes the sine/cosine component columns.
    
    Args:
        df (pd.DataFrame): The input DataFrame with sine/cosine components.
        save_dir (str): Directory where the cyclic scaling parameters are saved.
        
    Returns:
        pd.DataFrame: DataFrame with original cyclic columns restored.
    """
    df = df.copy()
    
    if not os.path.exists(save_dir):
        print(f"Save directory '{save_dir}' does not exist. Cannot undo cyclic scaling.")
        return df
    
    # Find all cyclic scaling files
    cyclic_files = [f for f in os.listdir(save_dir) if f.endswith('_cyclic.npz')]
    
    if not cyclic_files:
        print("No cyclic scaling files found to undo.")
        return df
    
    for cyclic_file in cyclic_files:
        # Extract column name from filename (remove '_cyclic.npz' suffix)
        col_name = cyclic_file[:-11]  # Remove '_cyclic.npz'
        sine_col = f"{col_name}_sine"
        cosine_col = f"{col_name}_cosine"
        
        # Check if both sine and cosine columns exist
        if sine_col not in df.columns or cosine_col not in df.columns:
            print(f"Warning: Sine/cosine columns for '{col_name}' not found. Skipping.")
            continue
        
        # Load the scaler
        scaler = CyclicScaler()
        scaler.load(f"{save_dir}/{cyclic_file}")
        
        # Reconstruct the sine/cosine data
        sin_cos_data = np.column_stack([df[sine_col], df[cosine_col]])
        
        # Unnormalize to get original values
        original_values = scaler.unnormalize(sin_cos_data)
        
        # Add the original column back
        df[col_name] = original_values
        
        # Remove the sine and cosine columns
        df.drop(columns=[sine_col, cosine_col], inplace=True)
        
        print(f"Restored original column '{col_name}' from sine/cosine components")
    
    return df


def redoCycleScale(df: pd.DataFrame, save_dir: str) -> pd.DataFrame:
    """
    Re-applies cyclic scaling using saved parameters.
    
    Looks for _cyclic.npz files in the save directory and reapplies the cyclic
    scaling transformations to the corresponding columns.
    
    Args:
        df (pd.DataFrame): The input DataFrame with original cyclic columns.
        save_dir (str): Directory where the cyclic scaling parameters are saved.
        
    Returns:
        pd.DataFrame: A copy of the DataFrame with cyclic scaling reapplied.
    """
    df_copy = df.copy()
    
    if not os.path.exists(save_dir):
        return df_copy
    
    # Find all cyclic scaling files
    cyclic_files = [f for f in os.listdir(save_dir) if f.endswith('_cyclic.npz')]
    
    if not cyclic_files:
        return df_copy
    
    for cyclic_file in cyclic_files:
        # Extract column name from filename
        col_name = cyclic_file[:-11]  # Remove '_cyclic.npz'
        
        if col_name not in df_copy.columns:
            print(f"Warning: Column '{col_name}' not found for cyclic scaling. Skipping.")
            continue
        
        # Load the scaler
        scaler = CyclicScaler()
        scaler.load(f"{save_dir}/{cyclic_file}")
        
        # Transform the data to sine/cosine components
        sin_cos_data = scaler.normalize(df_copy[col_name])
        
        # Add new columns for sine and cosine components
        df_copy[f"{col_name}_sine"] = sin_cos_data[:, 0]
        df_copy[f"{col_name}_cosine"] = sin_cos_data[:, 1]
        
        # Remove the original column
        df_copy.drop(columns=[col_name], inplace=True)
        
        print(f"Reapplied cyclic scaling to column '{col_name}'")
    
    return df_copy

    
def autonorm_dataframe(
    df,
    target_column: str,
    save_dir: str = "normalization_params",
    label_list: list = None,
    dummy_list: list = None,
    zscore_list: Optional[List[str]] = None,
    robust_list: Optional[List[str]] = None,
    minmax_list: Optional[List[str]] = None,
    find_replace_configs: Optional[List[FindReplaceConfig]] = None,
    cyclic_scale_configs: Optional[List[CyclicScaleConfig]] = None,
    feature_formulas: Optional[List[str]] = None,
    drop_columns: Optional[List[str]] = None
):
    """
    Normalizes a pandas DataFrame by applying various scaling methods
    (z-score, robust, min-max), encoding methods (label, dummy), and feature engineering
    techniques. It also allows for find and replace operations, cyclic scaling,
    and feature engineering formulas. The normalization parameters are saved to the specified directory.
    Args:
        df (pd.DataFrame): The input DataFrame to normalize.
        save_dir (str): Directory to save normalization parameters.
        label_list (list, optional): List of columns to label encode. Defaults to None.
        dummy_list (list, optional): List of columns to dummy encode. Defaults to None.
        zscore_list (list, optional): List of columns to z-score normalize. Defaults to None.
        robust_list (list, optional): List of columns to robust scale. Defaults to None.
        minmax_list (list, optional): List of columns to min-max scale. Defaults to None.
        find_replace_configs (list, optional): List of FindReplaceConfig objects for find and replace operations.
                                               Defaults to None.
        cyclic_scale_configs (list, optional): List of CyclicScaleConfig objects for cyclic scaling.
                                               Defaults to None.
        feature_formulas (list, optional): List of feature engineering formulas to apply. Defaults to None.
        drop_columns (list, optional): List of columns to drop from the DataFrame.
    Returns:
        pd.DataFrame: The normalized DataFrame.
    """

    df = df.copy()
    # build numerical column lists from the recommendations
    recommendations = recommend_scaling_methods(
        df, outlier_threshold=5.0, alpha=0.05, sample_limit=5000
    )
    
    if zscore_list is None:
        zscore_list = [
            col
            for col, rec in recommendations.items()
            if rec["should_scale"] and "z-score" in rec["recommended_method"].lower()
        ]
    if robust_list is None:
        robust_list = [
            col
            for col, rec in recommendations.items()
            if rec["should_scale"] and "robust" in rec["recommended_method"].lower()
        ]
    if minmax_list is None:
        minmax_list = [
            col
            for col, rec in recommendations.items()
            if rec["should_scale"] and "min-max" in rec["recommended_method"].lower()
        ]

    print("Columns to be robust scaled:", robust_list)
    # Print the recommendations


    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Do feature formulas if provided
    if feature_formulas:
        df = feature_engineer(df, feature_formulas, save_dir)

                
    # Do Cyclic scaling if a config is provided
    if cyclic_scale_configs:
        df = cycleScaleData(df, cyclic_scale_configs, save_dir)
                
    # Do Find/Replace if a config is provided
    if find_replace_configs:
        for config in find_replace_configs:
            df = find_and_replace(
                df,
                config.find_value,
                config.replace_value,
                config.column_list,
                save_dir,
            )
            
    
    print_recommendations(recommendations)
    print("Actual lists to be used for normalization:")
    print(f"Dummy List: {dummy_list if dummy_list else []}")
    print(f"Label List: {label_list if label_list else []}")
    print(f"Zscore List: {zscore_list}")
    print(f"Robust List: {robust_list}")
    print(f"MinMax List: {minmax_list}")
    # Normalize the DataFrame
    df = normalize_dataframe(
        df,
        save_dir,
        zscore_list=zscore_list,
        robust_list=robust_list,
        minmax_list=minmax_list,
        dummy_list=dummy_list,
        label_list=label_list,
    )

    # Drop specified columns if provided
    if drop_columns:
        for col in drop_columns:
            if col in df.columns:
                df.drop(columns=col, inplace=True)
                print(f"Dropped column: {col}")
            else:
                print(f"Column '{col}' not found in DataFrame. Skipping.")
                
        # Save the list of dropped columns to a JSON file
        dropped_columns_path = os.path.join(save_dir, "dropped_columns.json")
        with open(dropped_columns_path, "w") as f:
            json.dump(drop_columns, f, indent=4)


    # Ensure that the target_column is in the leftmost side of the DataFrame
    if target_column in df.columns:
        # Move the target column to the front
        cols = [target_column] + [col for col in df.columns if col != target_column]
        df = df[cols]
    else:
        print(f"Warning: Target column '{target_column}' not found in DataFrame. Skipping.")    
        
    # Get mapping for column name to index
    col_to_index = {col: idx for idx, col in enumerate(df.columns)}

    # Save the column mapping to a JSON file
    mapping_path = os.path.join(save_dir, "column_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(col_to_index, f, indent=4)
    print(f"Column mapping saved to {mapping_path}")
    # Print the column mapping
    print("Column mapping:")
    for col, idx in col_to_index.items():
        print(f"{col}: {idx}")

    return df

def undo_find_and_replace(df: pd.DataFrame, save_dir: str) -> pd.DataFrame:
    """
    Undoes all find and replace operations by applying the reverse transformations.

    Reads all find and replace mapping files and applies the reverse transformations
    in reverse order to undo the operations.

    Args:
        df (pd.DataFrame): The input DataFrame.
        save_dir (str): Directory where the mapping information is saved.

    Returns:
        pd.DataFrame: DataFrame with find and replace operations undone.
    """
    df = df.copy()

    # Return the original if the save directory doesn't exist
    if not os.path.exists(save_dir):
        return df

    # Get all mapping files in the directory
    mapping_files = [
        f
        for f in os.listdir(save_dir)
        if f.startswith("find_replace_mapping_") and f.endswith(".json")
    ]

    if not mapping_files:
        return df

    # Sort files by name in reverse order to undo in reverse sequence
    mapping_files.sort(reverse=True)

    for mapping_file in mapping_files:
        mapping_path = os.path.join(save_dir, mapping_file)

        with open(mapping_path, "r") as file:
            mapping_info = json.load(file)

        for col, info in mapping_info.items():
            if col in df.columns:
                # Apply reverse transformation: replace_value → find_value
                find_val = info["find"]
                replace_val = info["replace"]
                
                # Handle null values specifically
                if pd.isna(find_val):
                    # If original was null, replace the replacement back to null
                    df[col] = df[col].map(
                        lambda x: find_val if x == replace_val else x
                    )
                else:
                    # Regular replacement: replace_val → find_val
                    if pd.isna(replace_val):
                        df[col] = df[col].map(
                            lambda x: find_val if pd.isna(x) else x
                        )
                    else:
                        df[col] = df[col].map(
                            lambda x: find_val if x == replace_val else x
                        )

    return df

def find_and_replace(
    df: pd.DataFrame,
    find_value: str,
    replace_value: str,
    column_list: list,
    save_dir: str,
) -> pd.DataFrame:
    """
    Find and replace values in specified columns using map function. Save the mapping and column list to a json file for reversal.
    Each mapping will be saved to a unique file name based on how many times this function has been called on the same DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame.
        find_value (str): The value to find in the DataFrame.
        replace_value (str): The value to replace the found value with.
        column_list (list): List of columns to apply the find and replace operation on.
        save_dir (str): Directory to save the mapping information.
    Returns:
        pd.DataFrame: The DataFrame with values replaced.

    """
    
    df = df.copy()

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Get a count of the mapping files existing already
    existing_files = [
        f
        for f in os.listdir(save_dir)
        if f.startswith("find_replace_mapping_") and f.endswith(".json")
    ]
    mapping_count = len(existing_files)
    mapping_filename = f"find_replace_mapping_{mapping_count}.json"
    mapping_path = os.path.join(save_dir, mapping_filename)

    # Initialize a dictionary to store the mapping
    mapping_info = {}

    for col in column_list:
        if col in df.columns:
            # Create a mapping for the find and replace operation
            mapping_info[col] = {"find": find_value, "replace": replace_value}
            
            # Handle null values specifically
            if pd.isna(find_value):
                df[col] = df[col].map(lambda x: replace_value if pd.isna(x) else x)
            else:
                df[col] = df[col].map(lambda x: replace_value if x == find_value else x)
        else:
            print(f"Column '{col}' not found in DataFrame. Skipping.")

    # Save the mapping information to a JSON file
    with open(mapping_path, "w") as file:
        json.dump(mapping_info, file, indent=4)

    print(f"Find and replace mapping saved to {mapping_path}")

    return df





def redo_dummy_encode(df, save_dir: str):
    """
    Re-applies the dummy encoding transformation using saved base category information.

    Reads the JSON file saved by dummy_encode_and_save to determine the base category
    for each original column, then applies dummy encoding to a copy of the dataframe.

    Args:
        df (pd.DataFrame): The input DataFrame.
        save_dir (str): Directory where the base category information is saved.

    Returns:
        pd.DataFrame: A copy of the DataFrame with dummy encoding reapplied.
    """
    df_copy = df.copy()
    json_path = os.path.join(save_dir, "dummy_encoded_columns.json")

    # Return the copy if the JSON file doesn't exist
    if not os.path.exists(json_path):
        return df_copy

    with open(json_path, "r") as file:
        dummy_info = json.load(file)

    # Get the columns that need to be dummy encoded
    columns_to_encode = [col for col in dummy_info.keys() if col in df_copy.columns]

    if columns_to_encode:
        # Apply dummy encoding with drop_first=True to match the original transformation
        df_copy = pd.get_dummies(df_copy, columns=columns_to_encode, drop_first=True)

    return df_copy


def reMinMaxData(df, save_dir: str):
    """
    Re-applies min-max normalization using saved scaler parameters.

    Loads the saved MinMax scaler parameters and applies the same transformation
    to a copy of the dataframe.

    Args:
        df (pd.DataFrame): The input DataFrame.
        save_dir (str): Directory where the scaler parameters are saved.

    Returns:
        pd.DataFrame: A copy of the DataFrame with min-max normalization reapplied.
    """
    df_copy = df.copy()

    for col in df_copy.columns:
        scaler_path = f"{save_dir}/{col}_minmax.npz"
        if os.path.exists(scaler_path):
            scaler = MinMax()
            scaler.load(scaler_path)
            df_copy[col] = scaler.normalize(df_copy[col])

    return df_copy


def reZscoreData(df, save_dir: str):
    """
    Re-applies z-score normalization using saved scaler parameters.

    Loads the saved ZScore scaler parameters and applies the same transformation
    to a copy of the dataframe.

    Args:
        df (pd.DataFrame): The input DataFrame.
        save_dir (str): Directory where the scaler parameters are saved.

    Returns:
        pd.DataFrame: A copy of the DataFrame with z-score normalization reapplied.
    """
    df_copy = df.copy()

    for col in df_copy.columns:
        scaler_path = f"{save_dir}/{col}_zscore.npz"
        if os.path.exists(scaler_path):
            scaler = ZScore()
            scaler.load(scaler_path)
            df_copy[col] = scaler.normalize(df_copy[col])

    return df_copy


def reRobustData(df, save_dir: str):
    """
    Re-applies robust scaling using saved scaler parameters.

    Loads the saved RobustScaler parameters and applies the same transformation
    to a copy of the dataframe.

    Args:
        df (pd.DataFrame): The input DataFrame.
        save_dir (str): Directory where the scaler parameters are saved.

    Returns:
        pd.DataFrame: A copy of the DataFrame with robust scaling reapplied.
    """
    df_copy = df.copy()

    for col in df_copy.columns:
        scaler_path = f"{save_dir}/{col}_robust.npz"
        if os.path.exists(scaler_path):
            scaler = RobustScaler()
            scaler.load(scaler_path)
            df_copy[col] = scaler.normalize(df_copy[col])

    return df_copy

def redo_find_and_replace(df: pd.DataFrame, save_dir: str) -> pd.DataFrame:
    """
    Re-applies all find and replace operations using saved mappings.

    Reads all find and replace mapping files and applies the transformations
    in the same order they were originally applied to a copy of the dataframe.

    Args:
        df (pd.DataFrame): The input DataFrame.
        save_dir (str): Directory where the mapping information is saved.

    Returns:
        pd.DataFrame: A copy of the DataFrame with find and replace operations reapplied.
    """
    df_copy = df.copy()

    # Return the copy if the save directory doesn't exist
    if not os.path.exists(save_dir):
        return df_copy

    # Get all mapping files in the directory
    mapping_files = [
        f
        for f in os.listdir(save_dir)
        if f.startswith("find_replace_mapping_") and f.endswith(".json")
    ]

    if not mapping_files:
        return df_copy

    # Sort files by name to apply in the original order (not reverse like undo)
    mapping_files.sort()

    for mapping_file in mapping_files:
        mapping_path = os.path.join(save_dir, mapping_file)

        with open(mapping_path, "r") as file:
            mapping_info = json.load(file)

        for col, info in mapping_info.items():
            if col in df_copy.columns:
                # Handle null values specifically
                find_val = info["find"]
                replace_val = info["replace"]
                
                if pd.isna(find_val):
                    df_copy[col] = df_copy[col].map(
                        lambda x: replace_val if pd.isna(x) else x
                    )
                else:
                    df_copy[col] = df_copy[col].map(
                        lambda x: replace_val if x == find_val else x
                    )

    return df_copy

def redo_dataframe(df: pd.DataFrame, save_dir: str):
    """
    Re-applies all saved transformations to a copy of the dataframe.

    This orchestration function calls all individual redo functions in the appropriate
    order to reapply all saved transformations. The order matches the typical
    preprocessing pipeline: numerical scaling first, then categorical encoding.

    Args:
        df (pd.DataFrame): The input DataFrame.
        save_dir (str): Directory where all transformation parameters are saved.

    Returns:
        pd.DataFrame: A copy of the DataFrame with all transformations reapplied.
    """
    df_copy = df.copy()

    # Apply feature engineering first
    df_copy = redo_feature_engineer(df_copy, save_dir)

    # Apply numerical scaling transformations
    df_copy = reMinMaxData(df_copy, save_dir)
    df_copy = reZscoreData(df_copy, save_dir)
    df_copy = reRobustData(df_copy, save_dir)

    # Apply categorical encoding transformations

    

    # Apply cyclic scaling transformations
    df_copy = redoCycleScale(df_copy, save_dir)
    
    # Apply find and replace transformations
    df_copy = redo_find_and_replace(df_copy, save_dir)
    
    # Load dropped column list
    dropped_columns_path = os.path.join(save_dir, "dropped_columns.json")
    if os.path.exists(dropped_columns_path):
        with open(dropped_columns_path, "r") as f:
            dropped_columns = json.load(f)
        # Drop the columns that were previously dropped
        df_copy.drop(columns=dropped_columns, errors="ignore", inplace=True)

    df_copy = redo_label_encode(df_copy, save_dir)
    df_copy = redo_dummy_encode(df_copy, save_dir)
    
    # Ensure the DataFrame is returned in its original order
    df_copy = rearrange_columns_for_inference(df_copy, save_dir)
    return df_copy


def plot_training_metrics(
    checkpoint_dir: str,
    save_plots: bool = False,
    figsize: tuple = (12, 8),
    metrics_filename: str = "metrics.parquet",
    start_epoch: int = 0,
    end_epoch: Optional[int] = None,
) -> None:
    """
    Plot training metrics from checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory containing metrics.parquet
        save_plots: If True, save plots to checkpoint_dir. If False, display in notebook
        figsize: Figure size for each plot as (width, height)
        metrics_filename: Name of the metrics file (default: "metrics.parquet")
    """

    # Load metrics data
    metrics_path = os.path.join(checkpoint_dir, metrics_filename)

    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        return

    try:
        df = pd.read_parquet(metrics_path)
        
        # Filter by epoch range if specified
        if start_epoch > 0 or (end_epoch is not None and end_epoch < df['epoch'].max()):
            df = df[(df['epoch'] >= start_epoch) & (df['epoch'] <= (end_epoch if end_epoch is not None else df['epoch'].max()))]
    except Exception as e:
        print(f"Error loading metrics file: {e}")
        return

    if df.empty:
        print("Metrics dataframe is empty")
        return

    # Extract metric columns (exclude epoch and timestamp)
    exclude_cols = {"epoch", "timestamp"}
    metric_cols = [col for col in df.columns if col not in exclude_cols]

    if not metric_cols:
        print("No metric columns found in the dataframe")
        return

    # Group metrics by base name (accuracy, loss, etc.)
    metric_groups = {}
    for col in metric_cols:
        # Extract base metric name (e.g., 'accuracy' from 'train_accuracy')
        if col.startswith("train_"):
            base_name = col[6:]  # Remove 'train_' prefix
        elif col.startswith("test_"):
            base_name = col[5:]  # Remove 'test_' prefix
        else:
            base_name = col

        if base_name not in metric_groups:
            metric_groups[base_name] = []
        metric_groups[base_name].append(col)

    # Create plots for each metric group
    for metric_name, columns in metric_groups.items():
        plt.figure(figsize=figsize)

        # Plot each metric in the group
        for col in columns:
            if col in df.columns:
                # Determine line style and color
                if col.startswith("train_"):
                    label = f"Train {metric_name.title()}"
                    color = "blue"
                    linestyle = "-"
                elif col.startswith("test_"):
                    label = f"Test {metric_name.title()}"
                    color = "red"
                    linestyle = "--"
                else:
                    label = metric_name.title()
                    color = "green"
                    linestyle = "-"

                plt.plot(
                    df["epoch"],
                    df[col],
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    alpha=0.8,
                )

        # Customize plot
        plt.xlabel("Epoch", fontsize=12, fontweight="bold")
        plt.ylabel(metric_name.title(), fontsize=12, fontweight="bold")
        plt.title(f"{metric_name.title()} vs Epoch", fontsize=14, fontweight="bold")
        plt.legend(loc="best", frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)

        # Set integer ticks for epoch axis if reasonable number of epochs
        if len(df) <= 50:
            plt.xticks(df["epoch"])

        # Improve layout
        plt.tight_layout()

        # Save or display
        if save_plots:
            plot_filename = f"{metric_name}_vs_epoch.png"
            plot_path = os.path.join(checkpoint_dir, plot_filename)
            plt.savefig(
                plot_path,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            print(f"Saved plot: {plot_filename}")
            plt.close()  # Close to free memory when saving
        else:
            plt.show()

    # Create a summary plot with all metrics (if not saving individual plots)
    if not save_plots and len(metric_groups) > 1:
        _create_summary_plot(df, metric_groups, figsize, checkpoint_dir, save_plots)


def _create_summary_plot(
    df: pd.DataFrame,
    metric_groups: dict,
    figsize: tuple,
    checkpoint_dir: str,
    save_plots: bool,
) -> None:
    """Create a summary plot with all metrics in subplots."""

    n_metrics = len(metric_groups)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows)
    )

    # Handle single subplot case
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_cols > 1 else [axes]
    else:
        axes = axes.flatten()

    for idx, (metric_name, columns) in enumerate(metric_groups.items()):
        ax = axes[idx]

        for col in columns:
            if col in df.columns:
                if col.startswith("train_"):
                    label = f"Train {metric_name.title()}"
                    color = "blue"
                    linestyle = "-"
                elif col.startswith("test_"):
                    label = f"Test {metric_name.title()}"
                    color = "red"
                    linestyle = "--"
                else:
                    label = metric_name.title()
                    color = "green"
                    linestyle = "-"

                ax.plot(
                    df["epoch"],
                    df[col],
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2,
                    marker="o",
                    markersize=3,
                    alpha=0.8,
                )

        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel(metric_name.title(), fontweight="bold")
        ax.set_title(f"{metric_name.title()} vs Epoch", fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        if len(df) <= 50:
            ax.set_xticks(df["epoch"])

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Training Metrics Summary", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_plots:
        summary_path = os.path.join(checkpoint_dir, "metrics_summary.png")
        plt.savefig(
            summary_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        print(f"Saved summary plot: metrics_summary.png")
        plt.close()
    else:
        plt.show()


def list_available_metrics(
    checkpoint_dir: str, metrics_filename: str = "metrics.parquet"
) -> None:
    """
    List all available metrics in the checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory
        metrics_filename: Name of the metrics file
    """
    metrics_path = os.path.join(checkpoint_dir, metrics_filename)

    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        return

    try:
        df = pd.read_parquet(metrics_path)
        print(f"Available metrics in {metrics_path}:")
        print(f"Total epochs: {len(df)}")
        print(f"Columns: {list(df.columns)}")

        # Show sample data
        print("\nSample data:")
        print(df.head())

    except Exception as e:
        print(f"Error loading metrics file: {e}")


def plot_step_training_metrics(
    checkpoint_dir: str,
    save_plots: bool = False,
    figsize: tuple = (12, 8),
    metrics_filename: str = "metrics.parquet",
    start_step: int = 0,
    end_step: Optional[int] = None,
    smooth_window: int = 1,
) -> None:
    """
    Plot training metrics from train_flax_lm checkpoint directory with step-based x-axis.

    Args:
        checkpoint_dir: Path to checkpoint directory containing metrics.parquet
        save_plots: If True, save plots to checkpoint_dir. If False, display in notebook
        figsize: Figure size for each plot as (width, height)
        metrics_filename: Name of the metrics file (default: "metrics.parquet")
        start_step: Starting step for plotting (default: 0)
        end_step: Ending step for plotting (default: None for all steps)
        smooth_window: Window size for smoothing metrics (default: 1, no smoothing)
    """

    # Load metrics data
    metrics_path = os.path.join(checkpoint_dir, metrics_filename)

    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        return

    try:
        df = pd.read_parquet(metrics_path)
        
        # Filter by step range if specified
        if start_step > 0 or (end_step is not None and end_step < df['global_step'].max()):
            df = df[(df['global_step'] >= start_step) & (df['global_step'] <= (end_step if end_step is not None else df['global_step'].max()))]
            print(f"Filtered metrics to steps {start_step} to {end_step if end_step is not None else df['global_step'].max()}")
            print(f"Actual min/max global_step in filtered data: {df['global_step'].min()} to {df['global_step'].max()}")
            
        # Apply smoothing if specified
        if smooth_window > 1:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in ['global_step', 'epoch']:
                    df[col] = df[col].rolling(window=smooth_window, min_periods=1).mean()
                    
    except Exception as e:
        print(f"Error loading metrics file: {e}")
        return

    if df.empty:
        print("Metrics dataframe is empty")
        return

    # Extract metric columns (exclude global_step and epoch)
    exclude_cols = {"global_step", "epoch"}
    metric_cols = [col for col in df.columns if col not in exclude_cols]

    if not metric_cols:
        print("No metric columns found in the dataframe")
        return

    # Group metrics by base name (loss, etc.)
    metric_groups = {}
    for col in metric_cols:
        # Extract base metric name (e.g., 'loss' from 'train_loss')
        if col.startswith("train_"):
            base_name = col[6:]  # Remove 'train_' prefix
        elif col.startswith("test_"):
            base_name = col[5:]  # Remove 'test_' prefix
        else:
            base_name = col

        if base_name not in metric_groups:
            metric_groups[base_name] = []
        metric_groups[base_name].append(col)

    # Create plots for each metric group
    for metric_name, columns in metric_groups.items():
        plt.figure(figsize=figsize)

        # Plot each metric in the group
        for col in columns:
            if col in df.columns:
                # Skip NaN values for plotting
                valid_mask = ~pd.isna(df[col])
                if not valid_mask.any():
                    continue
                    
                x_data = df.loc[valid_mask, 'global_step']
                y_data = df.loc[valid_mask, col]
                
                # Determine line style and color
                if col.startswith("train_"):
                    label = f"Train {metric_name.title()}"
                    color = "blue"
                    linestyle = "-"
                    alpha = 0.8
                elif col.startswith("test_"):
                    label = f"Test {metric_name.title()}"
                    color = "red"
                    linestyle = "--"
                    alpha = 0.9
                else:
                    label = metric_name.title()
                    color = "green"
                    linestyle = "-"
                    alpha = 0.8

                plt.plot(
                    x_data,
                    y_data,
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2,
                    alpha=alpha,
                )

        # Customize plot
        plt.xlabel("Global Step", fontsize=12, fontweight="bold")
        plt.ylabel(metric_name.title(), fontsize=12, fontweight="bold")
        plt.title(f"{metric_name.title()} vs Global Step", fontsize=14, fontweight="bold")
        plt.legend(loc="best", frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)

        # Improve layout
        plt.tight_layout()

        # Save or display
        if save_plots:
            plot_filename = f"{metric_name}_vs_step.png"
            plot_path = os.path.join(checkpoint_dir, plot_filename)
            plt.savefig(
                plot_path,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            print(f"Saved plot: {plot_filename}")
            plt.close()  # Close to free memory when saving
        else:
            plt.show()

    # Create a summary plot with all metrics (if not saving individual plots)
    if not save_plots and len(metric_groups) > 1:
        _create_step_summary_plot(df, metric_groups, figsize, checkpoint_dir, save_plots)


def _create_step_summary_plot(
    df: pd.DataFrame,
    metric_groups: dict,
    figsize: tuple,
    checkpoint_dir: str,
    save_plots: bool,
) -> None:
    """Create a summary plot with all metrics in subplots using global_step as x-axis."""

    n_metrics = len(metric_groups)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows)
    )

    # Handle single subplot case
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_cols > 1 else [axes]
    else:
        axes = axes.flatten()

    for idx, (metric_name, columns) in enumerate(metric_groups.items()):
        ax = axes[idx]

        for col in columns:
            if col in df.columns:
                # Skip NaN values for plotting
                valid_mask = ~pd.isna(df[col])
                if not valid_mask.any():
                    continue
                    
                x_data = df.loc[valid_mask, 'global_step']
                y_data = df.loc[valid_mask, col]
                
                if col.startswith("train_"):
                    label = f"Train {metric_name.title()}"
                    color = "blue"
                    linestyle = "-"
                    alpha = 0.8
                elif col.startswith("test_"):
                    label = f"Test {metric_name.title()}"
                    color = "red"
                    linestyle = "--"
                    alpha = 0.9
                else:
                    label = metric_name.title()
                    color = "green"
                    linestyle = "-"
                    alpha = 0.8

                ax.plot(
                    x_data,
                    y_data,
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2,
                    alpha=alpha,
                )

        ax.set_xlabel("Global Step", fontweight="bold")
        ax.set_ylabel(metric_name.title(), fontweight="bold")
        ax.set_title(f"{metric_name.title()} vs Global Step", fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Step-Based Training Metrics Summary", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_plots:
        summary_path = os.path.join(checkpoint_dir, "step_metrics_summary.png")
        plt.savefig(
            summary_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        print(f"Saved step summary plot: step_metrics_summary.png")
        plt.close()
    else:
        plt.show()


# Example usage functions
def plot_mnist_example():
    """Example usage for MNIST training results."""
    # Display plots in notebook
    plot_training_metrics("./mnist_checkpoints", save_plots=True)

    # Or save plots to files
    # plot_training_metrics("./mnist_checkpoints", save_plots=True)


def plot_custom_metrics():
    """Example with custom figure size and metrics filename."""
    plot_training_metrics(
        checkpoint_dir="./custom_checkpoints",
        save_plots=True,
        figsize=(10, 6),
        metrics_filename="training_metrics.parquet",
    )


def rearrange_columns_for_inference(df: pd.DataFrame, save_dir: str) -> pd.DataFrame:
    """
    Rearranges DataFrame columns to match the expected order for inference.

    This function reads the saved column mapping and reorders the DataFrame columns
    to match the original training data structure. It handles cases where some
    columns might be missing by filling with appropriate defaults.

    Args:
        df (pd.DataFrame): The input DataFrame to rearrange.
        save_dir (str): Directory where the column mapping is saved.

    Returns:
        pd.DataFrame: DataFrame with columns rearranged in the correct order.
    """
    mapping_path = os.path.join(save_dir, "column_mapping.json")

    if not os.path.exists(mapping_path):
        print(f"Column mapping file not found: {mapping_path}")
        return df

    with open(mapping_path, "r") as f:
        original_columns_mapping = json.load(f)

    # Sort columns by their original index
    expected_columns = sorted(
        original_columns_mapping, key=lambda x: original_columns_mapping[x]
    )

    # Check for missing columns and add them with appropriate defaults
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        print(f"Warning: Missing columns for inference: {missing_columns}")
        for col in missing_columns:
            # Add missing columns with default values (you might want to customize this)
            df[col] = 0  # or np.nan, depending on your use case

    # Check for extra columns
    extra_columns = set(df.columns) - set(expected_columns)
    if extra_columns:
        print(f"Warning: Extra columns found (will be dropped): {extra_columns}")
        df = df.drop(columns=extra_columns)

    # Reorder columns to match expected order
    df = df[expected_columns]

    print(f"Columns rearranged for inference. Expected order: {expected_columns}")
    return df


def prepare_dataframe_for_inference(df: pd.DataFrame, save_dir: str) -> pd.DataFrame:
    """
    Complete preprocessing pipeline to prepare a DataFrame for inference.

    This function applies all saved transformations and ensures the DataFrame
    is in the correct format expected by the trained model.

    Args:
        df (pd.DataFrame): The raw input DataFrame.
        save_dir (str): Directory where all transformation parameters are saved.

    Returns:
        pd.DataFrame: DataFrame ready for inference.
    """
    # Apply all transformations
    df_processed = redo_dataframe(df, save_dir)

    # Ensure columns are in the correct order
    df_processed = rearrange_columns_for_inference(df_processed, save_dir)

    return df_processed


def feature_engineer(
    df: pd.DataFrame, formulas: List[str], save_dir: str
) -> pd.DataFrame:
    """
    Creates new features from formulas on existing columns and saves the recipe.

    This function parses mathematical formulas to create new columns and saves
    the formulas to a JSON file for later reapplication or removal.

    Args:
        df (pd.DataFrame): The input DataFrame.
        formulas (List[str]): List of formula strings in format "new_col = expression"
                             Example: ["bathroom_per_bedroom = bathrooms / bedrooms"]
        save_dir (str): Directory to save the feature engineering recipe.

    Returns:
        pd.DataFrame: DataFrame with new engineered features added.

    Example:
        >>> df = feature_engineer(df, [
        ...     "bathroom_per_bedroom = bathrooms / bedrooms",
        ...     "area_per_bedroom = area / bedrooms",
        ...     "total_rooms = bedrooms + bathrooms",
        ...     "area_density = area / (bedrooms + bathrooms)"
        ... ], "feature_params")
    """
    import re

    df = df.copy()
    # Ensure the save directory exists  
    os.makedirs(save_dir, exist_ok=True)

    # Load existing feature recipes if they exist
    recipe_path = os.path.join(save_dir, "feature_engineering_recipes.json")
    if os.path.exists(recipe_path):
        with open(recipe_path, "r") as f:
            existing_recipes = json.load(f)
    else:
        existing_recipes = []

    # Define available functions
    available_functions = {
        "np",
        "abs",
        "min",
        "max",
        "round",
        "pow",
        "sqrt",
        "log",
        "log10",
        "exp",
        "sin",
        "cos",
        "tan",
    }

    # Parse and apply formulas
    new_recipes = []
    created_columns = []

    for formula in formulas:
        try:
            # Parse the formula: "new_column = expression"
            if "=" not in formula:
                print(
                    f"Warning: Invalid formula format '{formula}'. Expected 'new_col = expression'"
                )
                continue

            parts = formula.split("=", 1)
            if len(parts) != 2:
                print(f"Warning: Invalid formula format '{formula}'")
                continue

            new_col_name = parts[0].strip()
            expression = parts[1].strip()

            # Validate that the new column name is valid
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", new_col_name):
                print(
                    f"Warning: Invalid column name '{new_col_name}'. Must be a valid Python identifier."
                )
                continue

            # Extract column names from the expression
            # This regex finds potential column names (letters, numbers, underscores)
            potential_cols = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", expression)

            # Filter out function names and only keep actual column references
            referenced_cols = [
                col
                for col in potential_cols
                if col in df.columns and col not in available_functions
            ]

            # Check if all referenced columns exist in the DataFrame
            missing_cols = [
                col
                for col in potential_cols
                if col not in df.columns and col not in available_functions
            ]
            if missing_cols:
                print(
                    f"Warning: Formula '{formula}' references missing columns: {missing_cols}"
                )
                continue

            # Create a safe evaluation environment with DataFrame columns and functions
            eval_env = {}

            # Add DataFrame columns to environment
            for col in df.columns:
                if col in potential_cols:
                    eval_env[col] = df[col]

            # Add numpy functions for mathematical operations
            eval_env.update(
                {
                    "np": np,
                    "abs": abs,
                    "min": min,
                    "max": max,
                    "round": round,
                    "pow": pow,
                    "sqrt": np.sqrt,
                    "log": np.log,
                    "log10": np.log10,
                    "exp": np.exp,
                    "sin": np.sin,
                    "cos": np.cos,
                    "tan": np.tan,
                }
            )

            # Evaluate the expression safely
            try:
                result = eval(expression, {"__builtins__": {}}, eval_env)
                df[new_col_name] = result
                created_columns.append(new_col_name)

                # Save the recipe (only save actual column references, not function names)
                recipe = {
                    "column_name": new_col_name,
                    "formula": formula,
                    "expression": expression,
                    "referenced_columns": referenced_cols,
                }
               
                new_recipes.append(recipe)

                print(f"Created feature: {new_col_name}")

            except Exception as e:
                print(f"Error evaluating formula '{formula}': {str(e)}")
                continue

        except Exception as e:
            print(f"Error processing formula '{formula}': {str(e)}")
            continue

    # Save all recipes (existing + new) to the file
    all_recipes = existing_recipes + new_recipes
    with open(recipe_path, "w") as f:
        json.dump(all_recipes, f, indent=4)

    if created_columns:
        print(
            f"Feature engineering complete. Created {len(created_columns)} new features."
        )
        print(f"Recipes saved to: {recipe_path}")
    else:
        print("No new features were created.")

    return df


def undo_feature_engineer(df: pd.DataFrame, save_dir: str) -> pd.DataFrame:
    """
    Removes all engineered features created by feature_engineer.

    Args:
        df (pd.DataFrame): The input DataFrame.
        save_dir (str): Directory where the feature engineering recipes are saved.

    Returns:
        pd.DataFrame: DataFrame with engineered features removed.
    """

    df = df.copy()
    recipe_path = os.path.join(save_dir, "feature_engineering_recipes.json")

    if not os.path.exists(recipe_path):
        print("No feature engineering recipes found to undo.")
        return df

    with open(recipe_path, "r") as f:
        recipes = json.load(f)

    removed_columns = []
    for recipe in recipes:
        col_name = recipe["column_name"]
        if col_name in df.columns:
            df = df.drop(columns=[col_name])
            removed_columns.append(col_name)

    if removed_columns:
        print(f"Removed {len(removed_columns)} engineered features: {removed_columns}")
    else:
        print("No engineered features found to remove.")

    return df


def redo_feature_engineer(df: pd.DataFrame, save_dir: str) -> pd.DataFrame:
    """
    Re-applies all saved feature engineering recipes to a copy of the dataframe.

    Args:
        df (pd.DataFrame): The input DataFrame.
        save_dir (str): Directory where the feature engineering recipes are saved.

    Returns:
        pd.DataFrame: A copy of the DataFrame with engineered features reapplied.
    """
    df_copy = df.copy()
    recipe_path = os.path.join(save_dir, "feature_engineering_recipes.json")

    if not os.path.exists(recipe_path):
        return df_copy

    with open(recipe_path, "r") as f:
        recipes = json.load(f)

    created_columns = []
    for recipe in recipes:
        try:
            col_name = recipe["column_name"]
            expression = recipe["expression"]
            referenced_cols = recipe["referenced_columns"]

            # Check if all referenced columns exist
            missing_cols = [
                col for col in referenced_cols if col not in df_copy.columns
            ]
            if missing_cols:
                print(
                    f"Warning: Cannot recreate '{col_name}' - missing columns: {missing_cols}"
                )
                continue

            # Create evaluation environment
            eval_env = {}
            for col in referenced_cols:
                eval_env[col] = df_copy[col]

            # Add numpy functions
            eval_env.update(
                {
                    "np": np,
                    "abs": abs,
                    "min": min,
                    "max": max,
                    "round": round,
                    "pow": pow,
                    "sqrt": np.sqrt,
                    "log": np.log,
                    "log10": np.log10,
                    "exp": np.exp,
                    "sin": np.sin,
                    "cos": np.cos,
                    "tan": np.tan,
                }
            )

            # Evaluate and create the feature
            result = eval(expression, {"__builtins__": {}}, eval_env)
            df_copy[col_name] = result
            created_columns.append(col_name)

        except Exception as e:
            print(f"Error recreating feature '{recipe['column_name']}': {str(e)}")
            continue

    if created_columns:
        print(
            f"Recreated {len(created_columns)} engineered features: {created_columns}"
        )

    return df_copy


def list_feature_recipes(save_dir: str) -> None:
    """
    Lists all saved feature engineering recipes.

    Args:
        save_dir (str): Directory where the feature engineering recipes are saved.
    """
    recipe_path = os.path.join(save_dir, "feature_engineering_recipes.json")

    if not os.path.exists(recipe_path):
        print("No feature engineering recipes found.")
        return

    with open(recipe_path, "r") as f:
        recipes = json.load(f)

    if not recipes:
        print("No feature engineering recipes found.")
        return

    print(f"Found {len(recipes)} feature engineering recipes:")
    print("-" * 50)

    for i, recipe in enumerate(recipes, 1):
        print(f"{i}. Column: {recipe['column_name']}")
        print(f"   Formula: {recipe['formula']}")
        print(f"   References: {recipe['referenced_columns']}")
        print()
