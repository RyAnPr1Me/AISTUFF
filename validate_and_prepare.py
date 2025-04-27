#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import warnings
import re
import traceback
from typing import Dict, List, Optional, Tuple, Union

#========================================================================
# Enhanced Data Validation and Cleaning Script with Advanced Features
#========================================================================

def setup_logging():
    """Configure detailed logging with timestamps"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Suppress specific warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', message='.*DataFrame.fillna with map-like replacements.*')


def try_format_dataframe(df, required_columns):
    """
    Attempt to coerce/format a DataFrame to the expected format with enhanced matching.
    - Try to extract or combine columns for 'text' and 'label'
    - Try to infer label from other columns if not present
    - Try to create a 'text' column from multiple possible text fields
    - Lowercase all column names for easier matching
    - Remove columns with all NaN or empty values
    Returns a new DataFrame or None if not possible.
    """
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.dropna(axis=1, how='all')  # Remove all-NaN columns
    
    # Extended search for text column
    if 'text' not in df.columns:
        text_candidates = [
            'headline', 'news', 'sentence', 'content', 'body', 'title', 
            'summary', 'description', 'text_field', 'article', 'message'
        ]
        # Exact matches first
        for c in text_candidates:
            if c in df.columns:
                df['text'] = df[c]
                logging.info(f"Using '{c}' column as 'text'")
                break
        
        # If no exact match, try substring matches
        if 'text' not in df.columns:
            for col in df.columns:
                for candidate in text_candidates:
                    if candidate in col:
                        df['text'] = df[col]
                        logging.info(f"Using '{col}' column as 'text' (substring match)")
                        break
                if 'text' in df.columns:
                    break
        
        # If still no matches, try to combine text fields
        if 'text' not in df.columns:
            combos = [c for c in df.columns if any(candidate in c for candidate in text_candidates)]
            if combos:
                df['text'] = df[combos].astype(str).agg(' '.join, axis=1)
                logging.info(f"Created 'text' by combining columns: {combos}")

    # Extended search for label column
    if 'label' not in df.columns:
        label_candidates = ['target', 'class', 'y', 'output', 'sentiment', 'category', 'prediction', 'result']
        
        # Exact matches first
        for c in label_candidates:
            if c in df.columns:
                df['label'] = df[c]
                logging.info(f"Using '{c}' column as 'label'")
                break
                
        # Try substring matches
        if 'label' not in df.columns:
            for col in df.columns:
                for candidate in label_candidates:
                    if candidate in col:
                        df['label'] = df[col]
                        logging.info(f"Using '{col}' column as 'label' (substring match)")
                        break
                if 'label' in df.columns:
                    break
        
        # Try to infer from financial data
        if 'label' not in df.columns:
            for c in df.columns:
                if 'return' in c:
                    df['label'] = (df[c] > 0).astype(int)
                    logging.info(f"Created binary 'label' from '{c}' (positive returns = 1)")
                    break
            
            if 'label' not in df.columns and 'future_close' in df.columns and 'close' in df.columns:
                df['label'] = (df['future_close'] > df['close']).astype(int)
                logging.info(f"Created binary 'label' from price movement (future_close > close)")

    # Collect financial and feature columns
    financial_columns = [
        'open', 'high', 'low', 'close', 'volume', 'adj_close', 'adj close',
        'sma', 'ema', 'macd', 'rsi', 'bb_', 'bollinger', 'stoch', 
        'volatility', 'atr', 'obv', 'roc'
    ]
    
    keep = [col for col in required_columns if col in df.columns]
    feature_cols = []
    
    # Find all feature columns based on prefixes or financial terms
    for c in df.columns:
        c_lower = c.lower()
        if (c_lower.startswith('feature_') or 
            any(term in c_lower for term in financial_columns) or
            c_lower.startswith('f_')):
            feature_cols.append(c)
    
    keep += feature_cols
    
    # Keep date column if present
    date_columns = ['date', 'datetime', 'timestamp', 'time']
    for date_col in date_columns:
        if date_col in df.columns and date_col not in keep:
            keep.append(date_col)
            break
    
    keep = list(dict.fromkeys(keep))  # Remove duplicates while preserving order
    if not all(col in keep for col in required_columns):
        missing = [col for col in required_columns if col not in keep]
        logging.error(f"Missing required columns after formatting: {missing}")
        return None
        
    df = df[keep]
    
    # Standardize column names
    df = df.rename(columns={col: col.lower() for col in df.columns})
    return df


def clean_text_columns(df: pd.DataFrame, text_cols: List[str]) -> pd.DataFrame:
    """
    Advanced text cleaning for multiple text columns.
    - Removes HTML/XML tags
    - Removes URLs
    - Removes special characters
    - Fixes common encoding issues
    - Normalizes whitespace
    """
    df = df.copy()
    
    for col in text_cols:
        if col in df.columns:
            # Convert to string and replace None/NaN
            df[col] = df[col].astype(str).replace('None', '').replace('nan', '')
            
            # Remove HTML tags
            df[col] = df[col].apply(lambda x: re.sub(r'<[^>]+>', '', x))
            
            # Remove URLs
            df[col] = df[col].apply(lambda x: re.sub(r'http\S+', '', x))
            
            # Fix common encoding issues
            df[col] = df[col].apply(lambda x: x.replace('&amp;', '&')
                                             .replace('&lt;', '<')
                                             .replace('&gt;', '>')
                                             .replace('&quot;', '"')
                                             .replace('&#39;', "'"))
            
            # Normalize whitespace
            df[col] = df[col].apply(lambda x: ' '.join(x.split()))
            
            # Strip and remove empty strings
            df[col] = df[col].str.strip()
    
    # Drop rows where all text columns are empty
    if text_cols:
        df = df[~df[text_cols].apply(lambda row: all(val == '' for val in row), axis=1)]
        
    return df


def detect_and_handle_anomalies(df: pd.DataFrame, 
                               numeric_cols: List[str],
                               contamination: float = 0.05) -> pd.DataFrame:
    """
    Detect anomalies using Isolation Forest and either:
    - Remove anomalies
    - Mark anomalies with a new column
    
    Parameters:
        df: Input DataFrame
        numeric_cols: List of numeric columns to use for anomaly detection
        contamination: Expected proportion of outliers (default 0.05)
    
    Returns:
        DataFrame with anomalies handled
    """
    if not numeric_cols or len(df) < 10:
        logging.warning("Not enough data or numeric columns for anomaly detection")
        return df
    
    try:
        # Make a copy of the data with just numeric columns for anomaly detection
        X = df[numeric_cols].copy()
        
        # Fill missing values with column means for anomaly detection
        X = X.fillna(X.mean())
        
        # If there are still NaNs (all NaN columns), drop those columns
        X = X.dropna(axis=1)
        
        if X.empty or X.shape[1] == 0:
            logging.warning("No valid numeric data for anomaly detection")
            return df
        
        # Apply Isolation Forest
        iso = IsolationForest(contamination=contamination, random_state=42)
        anomalies = iso.fit_predict(X)
        
        # Create anomaly flag column
        df['is_anomaly'] = np.where(anomalies == -1, True, False)
        
        # Log anomaly statistics
        anomaly_count = df['is_anomaly'].sum()
        logging.info(f"Detected {anomaly_count} anomalies ({anomaly_count/len(df)*100:.2f}% of data)")
        
        return df
        
    except Exception as e:
        logging.warning(f"Error in anomaly detection: {str(e)}")
        return df


def augment_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augment DataFrame with time-based features if date column exists
    """
    date_cols = [col for col in df.columns if col.lower() in ('date', 'datetime', 'timestamp')]
    
    if not date_cols:
        return df
        
    date_col = date_cols[0]
    
    try:
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
        # Drop rows with invalid dates
        df = df.dropna(subset=[date_col])
        
        if len(df) == 0:
            logging.warning("No valid dates found")
            return df
            
        # Extract time features
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['month'] = df[date_col].dt.month
        df['year'] = df[date_col].dt.year
        df['quarter'] = df[date_col].dt.quarter
        
        # Fiscal year features (assuming fiscal year starts in October)
        df['fiscal_quarter'] = ((df[date_col].dt.month - 10) % 12 // 3 + 1)
        
        # Is end of month
        df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
        
        # Is near holidays - US market holidays as example
        # This is a simplified approach - a more complete implementation would use a calendar library
        def is_near_holiday(date):
            # Major US holidays (simplified)
            year = date.year
            holidays = [
                # New Year's
                datetime(year, 1, 1),
                # MLK Day (3rd Monday in January)
                datetime(year, 1, 15) + timedelta(days=(7 - datetime(year, 1, 15).weekday()) % 7),
                # Presidents Day (3rd Monday in February)
                datetime(year, 2, 15) + timedelta(days=(7 - datetime(year, 2, 15).weekday()) % 7),
                # Good Friday (simplified as April 1)
                datetime(year, 4, 1),
                # Memorial Day (last Monday in May)
                datetime(year, 5, 31) - timedelta(days=datetime(year, 5, 31).weekday()),
                # Independence Day
                datetime(year, 7, 4),
                # Labor Day (1st Monday in September)
                datetime(year, 9, 1) + timedelta(days=(7 - datetime(year, 9, 1).weekday()) % 7),
                # Thanksgiving (4th Thursday in November)
                datetime(year, 11, 1) + timedelta(days=(3 - datetime(year, 11, 1).weekday()) % 7 + 21),
                # Christmas
                datetime(year, 12, 25)
            ]
            
            # Check if date is within 3 days of any holiday
            for holiday in holidays:
                if abs((date - holiday).days) <= 3:
                    return 1
            return 0
        
        # Apply holiday detection
        df['near_holiday'] = df[date_col].apply(is_near_holiday)
        
        logging.info("Added time-based features from date column")
        
    except Exception as e:
        logging.warning(f"Error adding time features: {str(e)}")
        
    return df


def impute_missing_values(df: pd.DataFrame, numeric_cols: List[str], 
                         categorical_cols: List[str] = None) -> pd.DataFrame:
    """
    Impute missing values in numeric and categorical columns
    - Uses KNN imputation for numeric columns
    - Uses mode imputation for categorical columns
    """
    if not numeric_cols:
        return df
        
    df_imputed = df.copy()
    
    # Handle numeric columns with KNN imputation
    if numeric_cols and any(df[numeric_cols].isna().any()):
        try:
            # Get only numeric columns with missing values
            cols_with_na = [col for col in numeric_cols if df[col].isna().any()]
            
            if cols_with_na:
                imputer = KNNImputer(n_neighbors=5)
                df_imputed[cols_with_na] = imputer.fit_transform(df[cols_with_na])
                logging.info(f"Imputed missing values in {len(cols_with_na)} numeric columns using KNN")
        except Exception as e:
            logging.warning(f"Error in KNN imputation: {str(e)}")
            # Fall back to mean imputation
            for col in numeric_cols:
                if df[col].isna().any():
                    df_imputed[col] = df[col].fillna(df[col].mean())
    
    # Handle categorical columns with mode imputation
    if categorical_cols:
        for col in categorical_cols:
            if col in df.columns and df[col].isna().any():
                df_imputed[col] = df[col].fillna(df[col].mode().iloc[0])
                
    return df_imputed


def validate_and_clean(df: pd.DataFrame,
                       required_columns: list,
                       tokenizer: AutoTokenizer,
                       advanced_cleaning: bool = True,
                       anomaly_detection: bool = True) -> pd.DataFrame:
    """
    Enhanced, adaptable validation and cleaning with advanced options:
    - Ensure required columns are present (case-insensitive, flexible mapping)
    - Advanced text cleaning for NLP
    - Anomaly detection and handling
    - Time feature augmentation
    - Missing value imputation
    - Test tokenization on samples
    - Detailed logging and diagnostics
    
    Returns cleaned DataFrame or None if validation fails.
    """
    if df is None or df.empty:
        logging.error("Received empty DataFrame")
        return None
        
    # Start with basic column mapping
    col_map = {c.lower(): c for c in df.columns}
    required_map = {}
    synonyms = {
        'text': ['text', 'headline', 'news', 'sentence', 'content', 'body', 'title', 'summary'],
        'label': ['label', 'target', 'class', 'y', 'output']
    }

    # Try to find required columns
    for req in required_columns:
        req_lower = req.lower()
        found = None
        
        # Direct match first
        if req_lower in col_map:
            found = col_map[req_lower]
        else:
            # Try synonyms
            for syn in synonyms.get(req_lower, []):
                if syn in col_map:
                    found = col_map[syn]
                    break
                    
        if found:
            required_map[req] = found
        else:
            logging.warning(f"Column '{req}' not found, attempting to format DataFrame...")
            df = try_format_dataframe(df, required_columns)
            if df is not None and req in df.columns:
                required_map[req] = req
            else:
                logging.error(f"Missing required column: '{req}' after formatting attempt.")
                return None

    # Rename columns to standard names
    df = df.rename(columns={v: k for k, v in required_map.items()})
    
    # Record initial stats
    initial_shape = df.shape
    logging.info(f"Initial data: {initial_shape[0]} rows, {initial_shape[1]} columns")
    
    # Remove rows with missing required values
    df.dropna(subset=required_columns, inplace=True)
    logging.info(f"After dropping rows with missing required values: {len(df)} rows")
    
    if len(df) == 0:
        logging.error("No data remaining after removing rows with missing required values")
        return None
    
    # Advanced text cleaning
    if advanced_cleaning and 'text' in df.columns:
        text_cols = [col for col in df.columns if col in ['text'] or 'text' in col.lower()]
        df = clean_text_columns(df, text_cols)
        logging.info(f"After text cleaning: {len(df)} rows")
    
    # Process labels
    if 'label' in df.columns:
        if df['label'].dtype == object:
            # Try to convert string labels to numeric
            try:
                df['label'] = pd.to_numeric(df['label'], errors='coerce')
            except Exception:
                # If conversion fails, map unique values to integers
                unique_values = df['label'].dropna().unique()
                label_map = {val: idx for idx, val in enumerate(unique_values)}
                df['label'] = df['label'].map(label_map)
                logging.info(f"Mapped labels: {label_map}")
        
        # Ensure labels are integers for classification
        if df['label'].dtype != 'int64':
            df['label'] = df['label'].astype('int64')
            
        # Check distribution
        label_counts = df['label'].value_counts()
        logging.info(f"Label distribution: {label_counts.to_dict()}")
        
        # Ensure at least two classes for binary classification
        if len(label_counts) < 2:
            logging.warning(f"Only one label value found: {label_counts.index[0]}")
    
    # Handle numeric columns with outlier detection and imputation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'label']  # Exclude label from numeric processing
    
    if numeric_cols:
        # Find and handle outliers
        if anomaly_detection:
            df = detect_and_handle_anomalies(df, numeric_cols)
            # Remove detected anomalies
            if 'is_anomaly' in df.columns:
                df = df[~df['is_anomaly']].drop('is_anomaly', axis=1)
                logging.info(f"After removing anomalies: {len(df)} rows")
        
        # Impute missing values
        df = impute_missing_values(df, numeric_cols)
        
        # Z-score filtering for extreme values
        zscores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / (df[numeric_cols].std(ddof=0) + 1e-8))
        extreme_mask = (zscores > 10).any(axis=1)  # Very high threshold for extreme outliers only
        if extreme_mask.any():
            df = df[~extreme_mask]
            logging.info(f"Removed {extreme_mask.sum()} rows with extreme values (z-score > 10)")
    
    # Augment with time features if date column exists
    df = augment_time_features(df)
    
    # Test tokenizer compatibility on sample texts
    if 'text' in df.columns and tokenizer is not None:
        sample_texts = df['text'].iloc[:min(5, len(df))].tolist()
        try:
            tokenizer(sample_texts, padding=True, truncation=True, return_tensors='pt')
            logging.info(f"Tokenizer successfully processed sample texts")
        except Exception as e:
            logging.error(f"Tokenizer error: {e}")
            return None
    
    # Remove duplicate rows
    before_dedup = len(df)
    df.drop_duplicates(subset=required_columns, inplace=True)
    if len(df) < before_dedup:
        logging.info(f"Removed {before_dedup - len(df)} duplicate rows")
    
    # Log processing summary
    logging.info(f"Data validation complete: {initial_shape[0]} -> {len(df)} rows ({len(df)/initial_shape[0]*100:.1f}% retained)")
    logging.info(f"Final columns: {', '.join(df.columns.tolist())}")
    
    return df


def analyze_dataset(df: pd.DataFrame) -> Dict:
    """Generate detailed dataset statistics and quality metrics"""
    stats = {}
    
    # Basic stats
    stats['row_count'] = len(df)
    stats['column_count'] = len(df.columns)
    stats['memory_usage'] = f"{df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB"
    
    # Data types
    type_counts = df.dtypes.value_counts().to_dict()
    stats['data_types'] = {str(k): int(v) for k, v in type_counts.items()}
    
    # Missing values
    missing = df.isna().sum()
    stats['missing_values'] = {col: int(count) for col, count in missing[missing > 0].items()}
    stats['total_missing'] = int(df.isna().sum().sum())
    stats['missing_percent'] = f"{df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.2f}%"
    
    # Text stats if present
    if 'text' in df.columns:
        text_lens = df['text'].astype(str).str.len()
        stats['text_length'] = {
            'min': int(text_lens.min()),
            'max': int(text_lens.max()),
            'mean': float(text_lens.mean()),
            'median': float(text_lens.median())
        }
        
        # Vocabulary size (approx)
        sample_size = min(1000, len(df))  # Limit sample for performance
        sample_texts = ' '.join(df['text'].sample(sample_size).astype(str))
        unique_words = set(re.findall(r'\w+', sample_texts.lower()))
        stats['vocab_size_estimate'] = len(unique_words)
        
        # Empty text count
        stats['empty_text_count'] = int((df['text'].astype(str).str.strip() == '').sum())
        
    # Label stats if present
    if 'label' in df.columns:
        label_counts = df['label'].value_counts().to_dict()
        stats['label_distribution'] = {str(k): int(v) for k, v in label_counts.items()}
        stats['class_imbalance_ratio'] = float(max(label_counts.values()) / min(label_counts.values()))
        
    # Feature stats for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'label']
    
    if numeric_cols.any():
        # Correlation with label if present
        if 'label' in df.columns:
            correlations = df[numeric_cols].corrwith(df['label']).abs().sort_values(ascending=False)
            stats['top_correlations'] = {col: float(corr) for col, corr in 
                                       list(correlations.items())[:10]}
    
    return stats


def inject_final_gaussian_noise(df, noise_std=0.01, noise_prob=0.2, exclude_cols=None):
    """
    Inject small Gaussian noise into numeric columns for a random subset of rows.
    Args:
        df: DataFrame to augment
        noise_std: Standard deviation of the Gaussian noise
        noise_prob: Probability of applying noise to each row
        exclude_cols: Columns to exclude from augmentation
    Returns:
        Augmented DataFrame
    """
    df_aug = df.copy()
    if exclude_cols is None:
        exclude_cols = []
    numeric_cols = df_aug.select_dtypes(include=[float, int, np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    if len(numeric_cols) == 0:
        return df_aug
    mask = np.random.rand(len(df_aug)) < noise_prob
    noise = np.random.normal(0, noise_std, size=(mask.sum(), len(numeric_cols)))
    df_aug.loc[mask, numeric_cols] += noise
    return df_aug


def main():
    setup_logging()
    
    logging.info("Data validation and preparation starting...")
    
    parser = argparse.ArgumentParser(
        description="Enhanced data validation, cleaning and preparation pipeline"
    )
    parser.add_argument(
        '--data-dir', type=str, default='Training_Data',
        help='Directory containing CSV files to validate'
    )
    parser.add_argument(
        '--output', type=str, default='Training_Data/validated_data.csv',
        help='Path for the combined validated CSV output'
    )
    parser.add_argument(
        '--tokenizer', type=str, default='bert-large-uncased',
        help='Hugging Face tokenizer model name'
    )
    parser.add_argument(
        '--columns', nargs='+', default=['text', 'label'],
        help='List of required columns in each CSV'
    )
    parser.add_argument(
        '--download-ticker', type=str, default=None,
        help='If set, download stock data for this ticker and create a dataset (CSV) in data-dir'
    )
    parser.add_argument(
        '--start', type=str, default='2015-01-01',
        help='Start date for stock data download (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end', type=str, default='2025-01-01',
        help='End date for stock data download (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--albert-only', action='store_true', default=True,
        help='If set, only process files prefixed with albert_ (default: True)'
    )
    # New arguments
    parser.add_argument(
        '--disable-advanced-cleaning', action='store_true',
        help='Disable advanced text cleaning'
    )
    parser.add_argument(
        '--disable-anomaly-detection', action='store_true',
        help='Disable anomaly detection'
    )
    parser.add_argument(
        '--report', action='store_true',
        help='Generate detailed dataset report'
    )
    
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        logging.error(f"Directory not found: {args.data_dir}")
        sys.exit(1)
    
    # Download stock data if requested
    if args.download_ticker:
        try:
            logging.info(f"Downloading stock data for {args.download_ticker}...")
            from download_and_prepare_stock_data import main as download_main
            
            # Set up sys.argv for download_main
            sys.argv = [
                'download_and_prepare_stock_data.py',
                '--symbol', args.download_ticker,
                '--start', args.start,
                '--end', args.end,
                '--output-dir', args.data_dir
            ]
            download_main()
        except Exception as e:
            logging.error(f"Failed to download stock data: {e}\n{traceback.format_exc()}")

    # Find and process CSV files
    csv_files = [f for f in os.listdir(args.data_dir) if f.lower().endswith('.csv')]
    
    # Only process ALBERT-formatted files unless --albert-only is False
    if args.albert_only:
        csv_files = [f for f in csv_files if f.startswith('albert_')]
        if not csv_files:
            logging.error("No ALBERT-formatted CSV files found. Run format_for_albert.py first.")
            logging.info("Available files: " + ", ".join(
                [f for f in os.listdir(args.data_dir) if f.lower().endswith('.csv')]))
            sys.exit(1)
    
    logging.info(f"Found {len(csv_files)} CSV files to process")

    # Initialize tokenizer once
    try:
        logging.info(f"Loading tokenizer: {args.tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    except Exception as e:
        logging.error(f"Failed to load tokenizer '{args.tokenizer}': {e}")
        sys.exit(1)

    validated_dfs = []
    stats_by_file = {}
    
    for fname in csv_files:
        path = os.path.join(args.data_dir, fname)
        logging.info(f"Processing {fname}...")
        
        try:
            df = pd.read_csv(path)
            logging.info(f"Loaded {len(df)} rows and {len(df.columns)} columns from {path}")
            
            cleaned = validate_and_clean(
                df, 
                args.columns, 
                tokenizer,
                advanced_cleaning=not args.disable_advanced_cleaning,
                anomaly_detection=not args.disable_anomaly_detection
            )
            
            if cleaned is not None and not cleaned.empty:
                validated_dfs.append(cleaned)
                
                # Generate stats if requested
                if args.report:
                    stats_by_file[fname] = analyze_dataset(cleaned)
            else:
                logging.warning(f"No valid data obtained from {fname}")
                
        except Exception as e:
            logging.error(f"Failed to process {fname}: {e}\n{traceback.format_exc()}")

    # Combine all valid dataframes
    if validated_dfs:
        combined = pd.concat(validated_dfs, ignore_index=True)
        
        # Final deduplication
        before_dedup = len(combined)
        combined = combined.drop_duplicates(subset=args.columns)
        if len(combined) < before_dedup:
            logging.info(f"Removed {before_dedup - len(combined)} duplicates from combined dataset")
        
        # Final step: inject small random noise into numeric features
        exclude_cols = args.columns if hasattr(args, 'columns') else ['text', 'label']
        combined = inject_final_gaussian_noise(combined, noise_std=0.01, noise_prob=0.2, exclude_cols=exclude_cols)
        logging.info("Injected small random Gaussian noise into numeric features as final augmentation step.")
        # Save
        combined.to_csv(args.output, index=False)
        logging.info(f"Saved {len(combined)} rows with {len(combined.columns)} columns to {args.output}")
        
        # Try to optimize data if possible
        try:
            optimize_path = args.output.replace('.csv', '_optimized.csv')
            logging.info(f"Optimizing data and saving to {optimize_path}...")
            
            from optimize_data import optimize_data_for_ai
            df_optimized = optimize_data_for_ai(combined)
            df_optimized.to_csv(optimize_path, index=False)
            logging.info(f"Saved optimized data: {len(df_optimized)} rows, {len(df_optimized.columns)} columns")
        except Exception as e:
            logging.warning(f"Could not optimize data: {e}")
            
        # Generate combined report if requested
        if args.report:
            report_path = args.output.replace('.csv', '_report.txt')
            with open(report_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write(f"DATASET QUALITY REPORT: {datetime.now()}\n")
                f.write("=" * 80 + "\n\n")
                
                # Overall stats
                overall_stats = analyze_dataset(combined)
                f.write("COMBINED DATASET SUMMARY\n")
                f.write("-" * 50 + "\n")
                for key, value in overall_stats.items():
                    if isinstance(value, dict):
                        f.write(f"{key}:\n")
                        for k, v in value.items():
                            f.write(f"  {k}: {v}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                f.write("\n\n")
                
                # Individual file stats
                f.write("INDIVIDUAL FILE STATISTICS\n")
                for fname, stats in stats_by_file.items():
                    f.write("-" * 50 + "\n")
                    f.write(f"File: {fname}\n")
                    f.write(f"Rows: {stats['row_count']}\n")
                    f.write(f"Columns: {stats['column_count']}\n")
                    if 'label_distribution' in stats:
                        f.write(f"Labels: {stats['label_distribution']}\n")
                    f.write("\n")
            
            logging.info(f"Generated dataset quality report: {report_path}")
    else:
        logging.error("No valid data after processing all files.")
        sys.exit(1)
    
    logging.info("Data validation and preparation complete!")

if __name__ == '__main__':
    main()

