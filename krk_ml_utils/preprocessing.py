import numpy as np
import yaml
import pandas as pd
import decimal
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

class MinMax:
    """Min-Max normalizer that scales data to range [0,1].
    This class implements min-max normalization which scales the data linearly
    between 0 and 1 by subtracting the minimum value and dividing by the range.
    Attributes:
        min: Minimum value in the training data
        max: Maximum value in the training data
    Methods:
        fit: Compute min and max values from training data
        normalize: Scale data to [0,1] range using stored min/max
        unnormalize: Reverse normalization to recover original scale
        save: Save min/max values to file
        load: Load min/max values from file
    Example:
        >>> scaler = MinMax()
        >>> scaler.fit(train_data)
        >>> normalized = scaler.normalize(data)
        >>> original = scaler.unnormalize(normalized)
    """

    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, data):
        self.min = np.min(data)
        self.max = np.max(data)

    def normalize(self, data):
        return (data - self.min) / (self.max - self.min)

    def unnormalize(self, data):
        return data * (self.max - self.min) + self.min

    # function to save the min and max values
    def save(self, filename):
        np.savez(filename, min=self.min, max=self.max)

    def load(self, filename):
        with np.load(filename) as data:
            self.min = data["min"]
            self.max = data["max"]


class ZScore:
    """Z-Score normalizer that scales data to have zero mean and unit variance.
    This class implements Z-score normalization (standardization) which scales the data
    to have a mean of 0 and a standard deviation of 1.
    Attributes:
        mean: Mean value of the training data
        std: Standard deviation of the training data
    Methods:
        fit: Compute mean and standard deviation from training data
        normalize: Scale data to zero mean and unit variance using stored mean/std
        unnormalize: Reverse normalization to recover original scale
        save: Save mean and standard deviation values to file
        load: Load mean and standard deviation values from file
    Example:
        >>> scaler = ZScore()
        >>> scaler.fit(train_data)
        >>> normalized = scaler.normalize(data)
        >>> original = scaler.unnormalize(normalized)
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)

    def normalize(self, data):
        return (data - self.mean) / self.std

    def unnormalize(self, data):
        return data * self.std + self.mean

    def save(self, filename):
        np.savez(filename, mean=self.mean, std=self.std)

    def load(self, filename):
        with np.load(filename) as data:
            self.mean = data["mean"]
            self.std = data["std"]


class RobustScaler:
    """Robust Scaler that scales data using median and IQR.
    This class implements robust scaling using median and Interquartile Range (IQR).
    It is less sensitive to outliers than MinMax or ZScore scaling.
    Attributes:
        median: Median value of the training data
        iqr: Interquartile Range of the training data
    Methods:
        fit: Compute median and IQR from training data
        normalize: Scale data using median and IQR
        unnormalize: Reverse normalization to recover original scale
        save: Save median and IQR values to file
        load: Load median and IQR values from file
    Example:
        >>> scaler = RobustScaler()
        >>> scaler.fit(train_data)
        >>> normalized = scaler.normalize(data)
        >>> original = scaler.unnormalize(normalized)
    """

    def __init__(self):
        self.median = None
        self.iqr = None

    def fit(self, data):
        self.median = np.median(data)
        q75, q25 = np.percentile(data, [75, 25])
        self.iqr = q75 - q25
        if self.iqr == 0:
            self.iqr = 1e-9  # avoid division by zero for constant data

    def normalize(self, data):
        return (data - self.median) / self.iqr

    def unnormalize(self, data):
        return data * self.iqr + self.median

    def save(self, filename):
        np.savez(filename, median=self.median, iqr=self.iqr)

    def load(self, filename):
        with np.load(filename) as data:
            self.median = data["median"]
            self.iqr = data["iqr"]


class CyclicScaler:
    """Cyclic Scaler that encodes cyclic data using sine and cosine transformations.
    
    This class implements cyclic encoding for periodic data (e.g., time of day, day of week,
    angles) by transforming scalar values into sine and cosine components. This preserves
    the cyclic nature of the data where the end value is close to the beginning value.
    
    Attributes:
        period: The period of the cyclic data (e.g., 24 for hours, 7 for days of week)
        
    Methods:
        fit: Store the period for the cyclic data
        normalize: Transform data to sine and cosine components
        unnormalize: Reverse transformation to recover original cyclic values
        save: Save period value to file
        load: Load period value from file
        
    Example:
        >>> scaler = CyclicScaler()
        >>> scaler.fit(hour_data, period=24)  # for hours 0-23
        >>> sin_cos = scaler.normalize(hour_data)  # returns [sin, cos] components
        >>> original = scaler.unnormalize(sin_cos)
    """
    
    def __init__(self):
        self.period = None
    
    def fit(self, data, period):
        """Fit the scaler with the period of the cyclic data.
        
        Args:
            data: Input data (not used for fitting, kept for API consistency)
            period: The period of the cyclic data
        """
        self.period = period
    
    def normalize(self, data):
        """Transform cyclic data to sine and cosine components.
        
        Args:
            data: Input cyclic data
            
        Returns:
            numpy array of shape (len(data), 2) with [sin, cos] components
        """
        if self.period is None:
            raise ValueError("Must call fit() before normalize()")
        
        # Convert to radians
        angle = 2 * np.pi * data / self.period
        
        # Return sine and cosine components
        return np.column_stack([np.sin(angle), np.cos(angle)])
    
    def unnormalize(self, data):
        """Recover original cyclic values from sine and cosine components.
        
        Args:
            data: Array of shape (n, 2) with [sin, cos] components
            
        Returns:
            Original cyclic values
        """
        if self.period is None:
            raise ValueError("Must call fit() before unnormalize()")
        
        sin_vals, cos_vals = data[:, 0], data[:, 1]
        
        # Recover angle using atan2
        angle = np.arctan2(sin_vals, cos_vals)
        
        # Convert back to original scale
        original = (angle * self.period) / (2 * np.pi)
        
        # Ensure values are in [0, period) range
        original = np.mod(original, self.period)
        
        return original
    
    def save(self, filename):
        """Save period to file."""
        np.savez(filename, period=self.period)
    
    def load(self, filename):
        """Load period from file."""
        with np.load(filename) as data:
            self.period = data["period"]



def to_p10_string(number, precision=3):
    """
    Converts a number into a P10 token-string representation with arbitrary precision.

    The P10 format breaks a number down into its scientific notation components:
    sign, mantissa, and exponent.
    Example: 32.5 with precision 3 -> 325 * 10^-1 -> '<+><3><2><5><E-1>'

    Args:
        number (int, float, str, decimal.Decimal): The number to convert.
        precision (int): The number of digits to include in the mantissa. Must be > 0.

    Returns:
        str: A concatenated string of P10 tokens representing the number.
        
    Raises:
        ValueError: If precision is not a positive integer.
    """
    if not isinstance(precision, int) or precision <= 0:
        raise ValueError("Precision must be a positive integer.")

    # --- 1. Handle the special case of zero ---
    # As per the paper's convention, 0 is handled as a special case.
    # We will represent it as a mantissa of all zeros with an exponent of 0.
    if decimal.Decimal(number).is_zero():
        sign_token = '<+>'
        mantissa_tokens = ''.join([f'<{d}>' for d in '0' * precision])
        exponent_token = '<E+0>'
        return f"{sign_token}{mantissa_tokens}{exponent_token}"

    # --- 2. Use Decimal for accurate arithmetic ---
    # Convert input to a Decimal object to avoid floating point inaccuracies.
    # The context sets the precision for rounding operations.
    ctx = decimal.Context(prec=precision)
    d = decimal.Decimal(number)

    # --- 3. Normalize to scientific notation ---
    # 'normalize()' converts the number to its scientific notation tuple:
    # (sign, digits, exponent). Example: Decimal('-123.45') -> (1, (1, 2, 3, 4, 5), -2)
    # The tuple format is (sign (0=positive, 1=negative), mantissa_digits, exponent)
    sign, digits, exponent = d.normalize(ctx).as_tuple()

    # --- 4. Construct Sign Token ---
    sign_token = '<->' if sign else '<+>'

    # --- 5. Construct Mantissa Tokens ---
    # The mantissa is formed by the first `precision` digits.
    # `len(digits)` might be less than `precision`, so we pad with zeros.
    # Example: 0.1 (prec=3) -> digits=(1,), needs to become '100'
    mantissa_str = ''.join(map(str, digits)).ljust(precision, '0')
    mantissa_tokens = ''.join([f'<{d}>' for d in mantissa_str])

    # --- 6. Adjust Exponent based on mantissa length ---
    # The exponent from as_tuple() assumes the decimal point is after the last digit.
    # We need to adjust it to match our normalized mantissa.
    # Example: 123.45 (prec=3) -> rounded to 123. Digits=(1,2,3), Exp=0. This is 123 * 10^0. Correct.
    # Example: 1234.5 (prec=3) -> rounded to 1230. Digits=(1,2,3), Exp=1. This is 123 * 10^1. Correct.
    # Example: 0.1234 (prec=3) -> rounded to 0.123. Digits=(1,2,3), Exp=-3. This is 123 * 10^-3. Correct.
    # The final exponent is the original exponent plus the number of digits minus the desired precision.
    final_exponent = exponent + len(digits) - precision

    # --- 7. Construct Exponent Token ---
    exp_sign = '+' if final_exponent >= 0 else '-'
    exponent_token = f'<E{exp_sign}{abs(final_exponent)}>'

    # --- 8. Combine and return ---
    return f"{sign_token}{mantissa_tokens}{exponent_token}"


def from_p10_string(p10_str: str) -> float:
    """
    Converts a P10 token-string back into a floating-point number.

    This function parses a string like '<+><1><2><3><E-1>' and reconstructs
    the number it represents.

    Args:
        p10_str (str): The concatenated P10 token string.

    Returns:
        float: The decoded floating-point number.
        
    Raises:
        ValueError: If the input string format is invalid.
    """
    # --- 1. Define a regular expression to parse the P10 string ---
    # This regex is designed to be robust and capture all components.
    # It looks for:
    #   ^<([+-])>       - Group 1: Sign token (+ or -) at the start.
    #   ((?:<\d>)+)      - Group 2: One or more digit tokens (e.g., '<1><2><3>').
    #                     (?:...) is a non-capturing group for the repeated digit token.
    #   <E([+-])(\d+)>$ - Group 3 & 4: Exponent token at the end.
    #                     Group 3 captures the exponent sign.
    #                     Group 4 captures the exponent value (one or more digits).
    pattern = re.compile(r"^<([+-])>((?:<\d>)+)<E([+-])(\d+)>$")
    match = pattern.match(p10_str)

    if not match:
        raise ValueError(f"Invalid P10 string format: '{p10_str}'")

    # --- 2. Extract components from the matched groups ---
    sign_char = match.group(1)
    mantissa_part = match.group(2)
    exp_sign_char = match.group(3)
    exp_value_str = match.group(4)

    # --- 3. Parse the Mantissa ---
    # Extract the digits from the mantissa tokens (e.g., '<1><2><3>' -> '123')
    mantissa_digits = "".join(re.findall(r"<(\d)>", mantissa_part))
    
    # --- 4. Parse the Sign and Exponent ---
    sign = 1 if sign_char == '+' else -1
    exponent_sign = 1 if exp_sign_char == '+' else -1
    exponent = int(exp_value_str)

    # --- 5. Reconstruct the number using Decimal for precision ---
    # The formula is: sign * mantissa * 10^(exponent_sign * exponent_value)
    # Using Decimal prevents potential floating point issues with large numbers.
    try:
        mantissa_val = decimal.Decimal(mantissa_digits)
        final_exponent = decimal.Decimal(exponent_sign * exponent)
        
        result_decimal = sign * mantissa_val * (decimal.Decimal(10) ** final_exponent)
    except decimal.InvalidOperation:
        raise ValueError("Error performing decimal arithmetic. Check mantissa/exponent values.")

    # --- 6. Return the result as a standard float ---
    return float(result_decimal)



def convert_for_text_to_text_regression(df, target_column, precision=10, drop_column_list: list = None):
    """    Convert a DataFrame for text-to-text regression tasks into a list of dictionaries suitable for YAML format
    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        target_column (str): The name of the target column to be used as output.
        drop_column_list (list, optional): List of columns to drop from the DataFrame. Defaults to None.    
    
    
    """

    

    # Drop specified columns if provided
    if drop_column_list is not None:
        df = df.drop(columns=drop_column_list, errors='ignore')
        
    # Convert dataframe to list of dictionaries
    data_list = df.to_dict(orient='records')
    
    processed_data = []
    for item in data_list:
        
        # Grab the target column if it exists in the dictionary
        target_value = item.get(target_column, None)
        if target_value is not None:
            target_value = to_p10_string(target_value, precision=precision)
            del item[target_column]  # Remove the original target column
        
        yaml_features = yaml.dump(item, allow_unicode=True, default_flow_style=False)
        
        processed_data.append({
            "input": yaml_features,
            "output": target_value
        })
        
    new_df = pd.DataFrame(processed_data)
    return new_df


def generate_p10_special_tokens(exponent_range: tuple = (-100, 100)) -> list:
    """
    Generates a list of all special tokens required for P10 encoding.
    
    Args:
        exponent_range (tuple): A (min, max) tuple for the exponent values.
    
    Returns:
        list: A list of special token strings.
    """
    special_tokens = []
    
    # 1. Sign tokens
    special_tokens.extend(['<+>', '<->'])
    
    # 2. Mantissa digit tokens (for constructing numbers of any precision)
    special_tokens.extend([f'<{i}>' for i in range(10)])
    
    # 3. Exponent tokens
    min_exp, max_exp = exponent_range
    for i in range(min_exp, max_exp + 1):
        sign = '+' if i >= 0 else '-'
        special_tokens.append(f'<E{sign}{abs(i)}>')
        
    return special_tokens


## Optimizations

def to_p10_string_vectorized(series, precision=3):
    """
    Vectorized version of to_p10_string for pandas Series.
    
    Args:
        series (pd.Series): Series of numbers to convert
        precision (int): Number of digits in mantissa
        
    Returns:
        pd.Series: Series of P10 token strings
    """
    def _to_p10_single(number):
        if pd.isna(number):
            return None
            
        if not isinstance(precision, int) or precision <= 0:
            raise ValueError("Precision must be a positive integer.")

        # Handle zero case
        if decimal.Decimal(str(number)).is_zero():
            sign_token = '<+>'
            mantissa_tokens = ''.join([f'<{d}>' for d in '0' * precision])
            exponent_token = '<E+0>'
            return f"{sign_token}{mantissa_tokens}{exponent_token}"

        # Use Decimal for accurate arithmetic
        ctx = decimal.Context(prec=precision)
        d = decimal.Decimal(str(number))
        
        # Normalize to scientific notation
        sign, digits, exponent = d.normalize(ctx).as_tuple()
        
        # Construct tokens
        sign_token = '<->' if sign else '<+>'
        mantissa_str = ''.join(map(str, digits)).ljust(precision, '0')
        mantissa_tokens = ''.join([f'<{d}>' for d in mantissa_str])
        
        final_exponent = exponent + len(digits) - precision
        exp_sign = '+' if final_exponent >= 0 else '-'
        exponent_token = f'<E{exp_sign}{abs(final_exponent)}>'
        
        return f"{sign_token}{mantissa_tokens}{exponent_token}"
    
    return series.apply(_to_p10_single)


def _fast_yaml_serialize(row_dict):
    """
    Fast YAML serialization for simple dictionaries.
    Falls back to yaml.dump for complex cases.
    """
    try:
        # For simple key-value pairs, build YAML manually (much faster)
        yaml_lines = []
        for key, value in row_dict.items():
            # Handle basic types efficiently
            if isinstance(value, (int, float)):
                yaml_lines.append(f"{key}: {value}")
            elif isinstance(value, str):
                # Simple string escaping
                if '\n' in value or '"' in value or "'" in value:
                    escaped_value = value.replace('"', '\\"')
                    yaml_lines.append(f'{key}: "{escaped_value}"')
                else:
                    yaml_lines.append(f"{key}: {value}")
            elif value is None:
                yaml_lines.append(f"{key}: null")
            else:
                # Fall back to yaml.dump for complex types
                return yaml.dump(row_dict, allow_unicode=True, default_flow_style=False)
        
        return '\n'.join(yaml_lines) + '\n'
    except Exception:
        # Fallback to original method
        return yaml.dump(row_dict, allow_unicode=True, default_flow_style=False)

def _process_chunk(chunk_data):
    """
    Process a chunk of data. This function needs to be at module level for multiprocessing.
    
    Args:
        chunk_data: tuple of (chunk_df, target_column, precision, drop_columns)
    
    Returns:
        pd.DataFrame: Processed chunk
    """
    chunk_df, target_column, precision, drop_columns = chunk_data
    
    # Make a copy to avoid modifying original
    df_chunk = chunk_df.copy()
    
    # Drop specified columns
    if drop_columns:
        df_chunk = df_chunk.drop(columns=drop_columns, errors='ignore')
    
    # Vectorized P10 conversion for target column
    if target_column in df_chunk.columns:
        target_values = to_p10_string_vectorized(df_chunk[target_column], precision=precision)
        df_chunk = df_chunk.drop(columns=[target_column])
    else:
        target_values = pd.Series([None] * len(df_chunk))
    
    # Convert to dictionaries and process
    processed_rows = []
    for idx, (_, row) in enumerate(df_chunk.iterrows()):
        row_dict = row.to_dict()
        yaml_features = _fast_yaml_serialize(row_dict)
        
        processed_rows.append({
            "input": yaml_features,
            "output": target_values.iloc[idx]
        })
    
    return pd.DataFrame(processed_rows)


def convert_for_text_to_text_regression_parallel(df, target_column, precision=10, drop_column_list: list = None, 
                                       chunk_size: int = 50000, n_workers: int = None):
    """
    Convert a DataFrame for text-to-text regression tasks with optimized performance.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        target_column (str): The name of the target column to be used as output.
        precision (int): Precision for P10 encoding. Defaults to 10.
        drop_column_list (list, optional): List of columns to drop from the DataFrame. Defaults to None.
        chunk_size (int): Size of chunks for processing. Defaults to 50000.
        n_workers (int, optional): Number of parallel workers. Defaults to CPU count - 1.
    
    Returns:
        pd.DataFrame: DataFrame with 'input' and 'output' columns.
    """
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    # For small DataFrames, use single-threaded processing
    if len(df) <= chunk_size:
        return _process_chunk((df, target_column, precision, drop_column_list))
    
    # Split DataFrame into chunks
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        chunks.append((chunk, target_column, precision, drop_column_list))
    
    # Process chunks in parallel
    if n_workers > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all chunks
            future_to_chunk = {executor.submit(_process_chunk, chunk): i 
                             for i, chunk in enumerate(chunks)}
            
            # Collect results in order
            results = [None] * len(chunks)
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    import datetime
                    start_time = datetime.datetime.now()
                    
                    results[chunk_idx] = future.result()
                    # Get chunk size from the original chunks data
                    chunk_df = chunks[chunk_idx][0]  # First element is the DataFrame
                    chunk_size_actual = len(chunk_df)
                    
                    print(f"Processing chunk of size {chunk_size_actual} at {start_time}")
                except Exception as exc:
                    print(f'Chunk {chunk_idx} generated an exception: {exc}')
                    # Process failed chunk sequentially as fallback
                    results[chunk_idx] = _process_chunk(chunks[chunk_idx])
            
            # Combine all results
            return pd.concat(results, ignore_index=True)
    else:
        # Single-threaded processing for chunks
        results = []
        for chunk_data in chunks:
            results.append(_process_chunk(chunk_data))
        return pd.concat(results, ignore_index=True)


# Alternative ultra-fast version for simple cases
def convert_for_text_to_text_regression_ultra_fast(df, target_column, precision=10, 
                                                  drop_column_list: list = None):
    """
    Ultra-fast version that trades some flexibility for maximum speed.
    Best for DataFrames with simple data types and no complex YAML requirements.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        target_column (str): The name of the target column to be used as output.
        precision (int): Precision for P10 encoding. Defaults to 10.
        drop_column_list (list, optional): List of columns to drop. Defaults to None.
    
    Returns:
        pd.DataFrame: DataFrame with 'input' and 'output' columns.
    """
    # Work with a copy
    df_work = df.copy()
    
    # Drop specified columns
    if drop_column_list:
        df_work = df_work.drop(columns=drop_column_list, errors='ignore')
    
    # Vectorized P10 conversion
    if target_column in df_work.columns:
        target_p10 = to_p10_string_vectorized(df_work[target_column], precision=precision)
        df_work = df_work.drop(columns=[target_column])
    else:
        target_p10 = pd.Series([None] * len(df_work))
    
    # Ultra-fast YAML generation using string operations
    def row_to_yaml_fast(row):
        yaml_parts = []
        for key, value in row.items():
            if pd.isna(value):
                yaml_parts.append(f"{key}: null")
            elif isinstance(value, (int, float)):
                yaml_parts.append(f"{key}: {value}")
            else:
                # Simple string handling
                value_str = str(value)
                if '\n' in value_str or '"' in value_str:
                    escaped_value = value_str.replace('"', '\\"')
                    yaml_parts.append(f'{key}: "{escaped_value}"')
                else:
                    yaml_parts.append(f"{key}: {value_str}")
        return '\n'.join(yaml_parts) + '\n'
    
    # Apply YAML conversion vectorized
    yaml_inputs = df_work.apply(row_to_yaml_fast, axis=1)
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'input': yaml_inputs,
        'output': target_p10
    })
    
    return result_df



### TOKENIZATION
def _tokenize_chunk(chunk_data):
    """Process a chunk of tokenization data."""
    texts, tokenizer, prefix, suffix = chunk_data
    
    # Add prefix/suffix to all texts in chunk
    formatted_texts = [f"{prefix}{text}{suffix}" for text in texts]
    
    # Tokenize the chunk
    tokens = [tokenizer.encode_as_ids(text) for text in formatted_texts]
    
    return tokens

def tokenize_columns_parallel(df, input_col="input", output_col="output", 
                            tokenizer=None, chunk_size=10000, n_workers=None):
    """
    Parallel tokenization for large DataFrames.
    
    Args:
        df: DataFrame with input/output columns
        input_col: Name of input column
        output_col: Name of output column
        tokenizer: SentencePiece tokenizer object
        chunk_size: Size of chunks for parallel processing
        n_workers: Number of parallel workers
    
    Returns:
        DataFrame with added token columns
    """
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    df_result = df.copy()
    
    # Process input and output columns separately
    for col_name, token_col_name in [(input_col, "input_tokens"), (output_col, "output_tokens")]:
        texts = df_result[col_name].astype(str).tolist()
        
        if len(texts) <= chunk_size:
            # Small dataset - process directly
            tokens = _tokenize_chunk((texts, tokenizer, "<s>", "</s>"))
            df_result[token_col_name] = tokens
        else:
            # Large dataset - process in parallel chunks
            chunks = []
            for i in range(0, len(texts), chunk_size):
                chunk_texts = texts[i:i + chunk_size]
                chunks.append((chunk_texts, tokenizer, "<s>", "</s>"))
            
            # Process chunks in parallel
            if n_workers > 1:
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    future_to_chunk = {executor.submit(_tokenize_chunk, chunk): i 
                                     for i, chunk in enumerate(chunks)}
                    
                    results = [None] * len(chunks)
                    for future in as_completed(future_to_chunk):
                        chunk_idx = future_to_chunk[future]
                        results[chunk_idx] = future.result()
                    
                    # Flatten results
                    all_tokens = []
                    for result in results:
                        all_tokens.extend(result)
                    
                    df_result[token_col_name] = all_tokens
            else:
                # Single-threaded chunk processing
                all_tokens = []
                for chunk in chunks:
                    tokens = _tokenize_chunk(chunk)
                    all_tokens.extend(tokens)
                df_result[token_col_name] = all_tokens
    
    return df_result

def tokenize_datasets_optimized(train_set, val_set, tokenizer, chunk_size=50000):
    """
    Optimized tokenization for train/val sets with progress tracking.
    
    Args:
        train_set: Training DataFrame
        val_set: Validation DataFrame  
        tokenizer: SentencePiece tokenizer
        chunk_size: Chunk size for processing
    
    Returns:
        tuple: (tokenized_train_set, tokenized_val_set)
    """
    import time
    
    def tokenize_single_dataset(df, dataset_name):
        print(f"Tokenizing {dataset_name} set ({len(df):,} rows)...")
        start_time = time.time()
        
        # Vectorized string concatenation
        input_texts = ("<s>" + df["input"].astype(str) + "</s>").tolist()
        output_texts = ("<s>" + df["output"].astype(str) + "</s>").tolist()
        
        # Batch tokenization with progress
        print(f"  Processing input tokens...")
        input_tokens = []
        for i in range(0, len(input_texts), chunk_size):
            chunk = input_texts[i:i + chunk_size]
            chunk_tokens = [tokenizer.encode_as_ids(text) for text in chunk]
            input_tokens.extend(chunk_tokens)
            print(f"    Processed {min(i + chunk_size, len(input_texts)):,}/{len(input_texts):,} inputs")
        
        print(f"  Processing output tokens...")
        output_tokens = []
        for i in range(0, len(output_texts), chunk_size):
            chunk = output_texts[i:i + chunk_size]
            chunk_tokens = [tokenizer.encode_as_ids(text) for text in chunk]
            output_tokens.extend(chunk_tokens)
            print(f"    Processed {min(i + chunk_size, len(output_texts)):,}/{len(output_texts):,} outputs")
        
        # Add to DataFrame
        df_result = df.copy()
        df_result["input_tokens"] = input_tokens
        df_result["output_tokens"] = output_tokens
        
        elapsed = time.time() - start_time
        print(f"  Completed {dataset_name} in {elapsed:.2f}s ({len(df)/elapsed:.0f} rows/sec)")
        
        return df_result
    
    # Process both datasets
    train_tokenized = tokenize_single_dataset(train_set, "train")
    val_tokenized = tokenize_single_dataset(val_set, "validation")
    
    return train_tokenized, val_tokenized


## TOKENIZATION OPTIMIZATIONS

def tokenize_columns_memory_efficient_parallel(df, input_col="input", output_col="output", 
                                             tokenizer=None, chunk_size=5000, n_workers=None):
    """
    Memory-efficient parallel tokenization that processes one column at a time.
    
    Args:
        df: DataFrame with input/output columns
        input_col: Name of input column
        output_col: Name of output column
        tokenizer: SentencePiece tokenizer object
        chunk_size: Size of chunks for parallel processing (smaller for memory efficiency)
        n_workers: Number of parallel workers
    
    Returns:
        DataFrame with added token columns
    """
    import gc
    import time
    
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    print(f"Starting memory-efficient parallel tokenization with {n_workers} workers...")
    print(f"Chunk size: {chunk_size:,}")
    
    df_result = df.copy()
    
    # Process each column separately to minimize memory usage
    for col_name, token_col_name in [(input_col, "input_tokens"), (output_col, "output_tokens")]:
        print(f"\nProcessing {col_name} column ({len(df):,} rows)...")
        start_time = time.time()
        
        # Get texts as list
        texts = df_result[col_name].astype(str).tolist()
        
        # Create chunks
        chunks = []
        for i in range(0, len(texts), chunk_size):
            chunk_texts = texts[i:i + chunk_size]
            chunks.append((chunk_texts, tokenizer, "<s>", "</s>"))
        
        print(f"  Created {len(chunks)} chunks")
        
        # Process chunks in parallel with memory management
        if n_workers > 1 and len(chunks) > 1:
            # Process in batches to control memory usage
            batch_size = max(1, n_workers * 2)  # Process 2x workers worth of chunks at a time
            all_tokens = [None] * len(chunks)
            
            for batch_start in range(0, len(chunks), batch_size):
                batch_end = min(batch_start + batch_size, len(chunks))
                batch_chunks = chunks[batch_start:batch_end]
                
                print(f"    Processing batch {batch_start//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    future_to_idx = {
                        executor.submit(_tokenize_chunk, chunk): batch_start + i 
                        for i, chunk in enumerate(batch_chunks)
                    }
                    
                    for future in as_completed(future_to_idx):
                        chunk_idx = future_to_idx[future]
                        try:
                            all_tokens[chunk_idx] = future.result()
                        except Exception as exc:
                            print(f'    Chunk {chunk_idx} failed: {exc}')
                            # Fallback to sequential processing for this chunk
                            all_tokens[chunk_idx] = _tokenize_chunk(chunks[chunk_idx])
                
                # Force garbage collection after each batch
                gc.collect()
                
                # Progress update
                processed_rows = min(batch_end * chunk_size, len(texts))
                print(f"    Completed {processed_rows:,}/{len(texts):,} rows")
            
            # Flatten results
            final_tokens = []
            for token_list in all_tokens:
                final_tokens.extend(token_list)
            
            df_result[token_col_name] = final_tokens
            
        else:
            # Single-threaded processing
            all_tokens = []
            for i, chunk in enumerate(chunks):
                tokens = _tokenize_chunk(chunk)
                all_tokens.extend(tokens)
                if i % 10 == 0:
                    processed = min((i + 1) * chunk_size, len(texts))
                    print(f"    Processed {processed:,}/{len(texts):,} rows")
            
            df_result[token_col_name] = all_tokens
        
        # Cleanup after each column
        del texts, chunks, all_tokens
        gc.collect()
        
        elapsed = time.time() - start_time
        print(f"  Completed {col_name} in {elapsed:.2f}s ({len(df)/elapsed:.0f} rows/sec)")
    
    return df_result


def tokenize_datasets_large_scale(train_set, val_set, tokenizer, 
                                chunk_size=5000, n_workers=None, 
                                save_checkpoints=True, checkpoint_dir="./checkpoints"):
    """
    Large-scale tokenization with checkpointing and memory management.
    
    Args:
        train_set: Training DataFrame
        val_set: Validation DataFrame
        tokenizer: SentencePiece tokenizer
        chunk_size: Chunk size (smaller for large datasets)
        n_workers: Number of workers
        save_checkpoints: Whether to save intermediate results
        checkpoint_dir: Directory for checkpoints
    
    Returns:
        tuple: (tokenized_train_set, tokenized_val_set)
    """
    import os
    import pickle
    import gc
    
    if save_checkpoints:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def process_dataset_with_checkpoints(df, dataset_name):
        checkpoint_file = os.path.join(checkpoint_dir, f"{dataset_name}_tokenized.pkl") if save_checkpoints else None
        
        # Check if checkpoint exists
        if checkpoint_file and os.path.exists(checkpoint_file):
            print(f"Loading {dataset_name} from checkpoint...")
            with open(checkpoint_file, 'rb') as f:
                return pickle.load(f)
        
        print(f"\nTokenizing {dataset_name} set ({len(df):,} rows)")
        
        # Process with memory-efficient parallel tokenization
        result = tokenize_columns_memory_efficient_parallel(
            df, 
            tokenizer=tokenizer, 
            chunk_size=chunk_size, 
            n_workers=n_workers
        )
        
        # Save checkpoint
        if checkpoint_file:
            print(f"Saving {dataset_name} checkpoint...")
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(result, f)
        
        # Force garbage collection
        gc.collect()
        
        return result
    
    # Process datasets sequentially to manage memory
    print("=== Processing Training Set ===")
    train_result = process_dataset_with_checkpoints(train_set, "train")
    
    # Force cleanup before processing validation set
    gc.collect()
    
    print("\n=== Processing Validation Set ===")
    val_result = process_dataset_with_checkpoints(val_set, "val")
    
    return train_result, val_result


# Ultra memory-efficient version for extremely large datasets
def tokenize_columns_streaming(df, input_col="input", output_col="output", 
                             tokenizer=None, chunk_size=1000, n_workers=None):
    """
    Streaming tokenization that processes data in very small chunks to minimize memory.
    Best for datasets that don't fit in memory.
    """
    import gc
    import time
    
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    print(f"Starting streaming tokenization with chunk size {chunk_size:,}")
    
    # Initialize result lists
    input_tokens = []
    output_tokens = []
    
    # Process in small batches
    for i in range(0, len(df), chunk_size):
        batch_df = df.iloc[i:i + chunk_size].copy()
        
        # Process this batch
        batch_result = tokenize_columns_memory_efficient_parallel(
            batch_df,
            input_col=input_col,
            output_col=output_col,
            tokenizer=tokenizer,
            chunk_size=min(500, chunk_size // 2),  # Even smaller chunks for parallel processing
            n_workers=min(n_workers, 2)  # Fewer workers to save memory
        )
        
        # Extend result lists
        input_tokens.extend(batch_result["input_tokens"].tolist())
        output_tokens.extend(batch_result["output_tokens"].tolist())
        
        # Cleanup
        del batch_df, batch_result
        gc.collect()
        
        # Progress
        processed = min(i + chunk_size, len(df))
        if i % (chunk_size * 10) == 0:
            print(f"Processed {processed:,}/{len(df):,} rows")
    
    # Create final result
    result_df = df.copy()
    result_df["input_tokens"] = input_tokens
    result_df["output_tokens"] = output_tokens
    
    return result_df