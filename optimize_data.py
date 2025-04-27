import argparse
import pandas as pd
import numpy as np
import logging
import os
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def optimize_data_for_ai(df, label_col='label', text_col='text', corr_thresh=0.95, min_var=1e-8, 
                         use_mutual_info=True, use_random_forest=True, feature_keep_percent=0.8,
                         generate_report=True):
    """
    Enhanced data optimization with multiple feature selection techniques:
    - Removes constant/low variance columns
    - Removes highly correlated features 
    - Uses mutual information to identify most predictive features
    - Uses Random Forest importance to select best features
    - Generates a feature importance report
    
    Args:
        df: Input DataFrame
        label_col: Name of the target column
        text_col: Name of the text column
        corr_thresh: Correlation threshold for removing correlated features
        min_var: Minimum variance threshold
        use_mutual_info: Whether to use mutual information for feature selection
        use_random_forest: Whether to use random forest importance for feature selection
        feature_keep_percent: Percentage of features to keep after selection
        generate_report: Whether to generate a feature importance report
        
    Returns:
        Optimized DataFrame
    """
    drop_cols = []
    original_shape = df.shape
    non_feature_cols = [label_col, text_col]
    
    # Keep a copy of the original dataframe for reporting
    df_original = df.copy()
    
    # Make sure the label column exists
    if label_col not in df.columns:
        logging.error(f"Label column '{label_col}' not found in data.")
        return df
    
    # Step 1: Basic cleaning - Remove constant columns
    nunique = df.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        drop_cols.extend([col for col in const_cols if col not in non_feature_cols])
        logging.info(f"Data optimizer: Dropping {len(drop_cols)} constant columns")

    # Step 2: Remove low variance columns
    var = df.var(numeric_only=True)
    low_var_cols = [col for col in var.index if var[col] < min_var and col not in non_feature_cols]
    if low_var_cols:
        drop_cols.extend(low_var_cols)
        logging.info(f"Data optimizer: Dropping {len(low_var_cols)} low-variance columns")

    # Step 3: Preprocess date columns first
    logging.info("Checking for date columns that need preprocessing...")
    date_cols = []
    
    # Identify potential date columns
    for col in df.columns:
        if col not in non_feature_cols:
            # Check sample values for date-like strings
            if df[col].dtype == 'object':
                sample_val = str(df[col].iloc[0]).lower() if not pd.isna(df[col].iloc[0]) else ""
                if '1970-01-01' in sample_val or \
                   any(x in col.lower() for x in ['date', 'time', 'day', 'month', 'year', 'timestamp']):
                    date_cols.append(col)
                    logging.info(f"Detected date column: {col}")
    
    # Process date columns
    for col in date_cols:
        try:
            # Convert to datetime
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
            if not df[col].isna().all():
                # Extract useful date features and drop original column
                df[f'{col}_day_of_week'] = df[col].dt.dayofweek
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                
                # Extract hour if time data exists
                if (df[col].dt.hour != 0).any():
                    df[f'{col}_hour'] = df[col].dt.hour
                
                # Is weekend
                df[f'{col}_is_weekend'] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
                
                # Drop original date column
                drop_cols.append(col)
                logging.info(f"Extracted features from date column '{col}' and marked for dropping")
            else:
                # If conversion entirely failed, mark for dropping
                drop_cols.append(col)
                logging.info(f"Date column '{col}' could not be converted, marking for dropping")
        except Exception as e:
            # If any error, just drop the column
            drop_cols.append(col)
            logging.warning(f"Error processing date column '{col}': {str(e)}")
    
    # Drop any remaining non-numeric columns that would cause problems
    remaining_cols = [col for col in df.columns if col not in drop_cols and col not in non_feature_cols]
    for col in remaining_cols:
        if df[col].dtype == 'object':
            try:
                # Try numeric conversion
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # If too many NaNs resulted, drop it
                if df[col].isna().sum() > 0.3 * len(df):
                    drop_cols.append(col)
                    logging.info(f"Dropping column '{col}' - too many NaNs after numeric conversion")
            except:
                drop_cols.append(col)
                logging.info(f"Dropping non-numeric column '{col}'")

    # Step 4: Remove highly correlated columns (keep only one from each group)
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    feature_cols = [col for col in feature_cols if col not in drop_cols]
    
    # Only run correlation analysis if we have enough features
    if len(feature_cols) > 1:
        logging.info("Analyzing feature correlations...")
        corr_matrix = df[feature_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find feature pairs with high correlation
        high_corr_pairs = []
        for col in upper.columns:
            high_corr_features = upper.index[upper[col] > corr_thresh].tolist()
            if high_corr_features:
                for feat in high_corr_features:
                    high_corr_pairs.append((col, feat, upper.loc[feat, col]))
        
        # Sort by correlation strength
        high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Iteratively remove highly correlated features
        corr_drop_candidates = set()
        for col1, col2, corr in high_corr_pairs:
            if col1 not in corr_drop_candidates and col2 not in corr_drop_candidates:
                # Decide which one to drop based on:
                # 1. Feature name (prefer to keep features with 'close' or fundamentals)
                # 2. Column name length (shorter names often more fundamental)
                if 'close' in col1.lower() or 'price' in col1.lower():
                    corr_drop_candidates.add(col2)
                elif 'close' in col2.lower() or 'price' in col2.lower():
                    corr_drop_candidates.add(col1)
                else:
                    # Drop the one with longer name as a heuristic
                    corr_drop_candidates.add(col2 if len(col1) <= len(col2) else col1)
        
        if corr_drop_candidates:
            drop_cols.extend(list(corr_drop_candidates))
            logging.info(f"Data optimizer: Dropping {len(corr_drop_candidates)} highly correlated columns")
    
    # Initial feature elimination
    df_filtered = df.drop(columns=drop_cols, errors='ignore')
    feature_cols = [col for col in df_filtered.columns if col not in non_feature_cols]
    X = df_filtered[feature_cols]
    
    # Ensure we have the label column as integers for feature selection
    y = df_filtered[label_col]
    if not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        y = le.fit_transform(y)
        
    feature_importances = {}
    kept_features = set(feature_cols)
    
    # Step 5: Mutual Information-based feature selection
    if use_mutual_info and len(feature_cols) > 5:
        try:
            logging.info("Running mutual information analysis...")
            # Handle potential categorical features by converting them to numeric
            X_numeric = X.copy()
            for col in X_numeric.columns:
                if not pd.api.types.is_numeric_dtype(X_numeric[col]):
                    X_numeric[col] = pd.factorize(X_numeric[col])[0]
            
            mi_scores = mutual_info_classif(X_numeric, y, random_state=42)
            mi_scores = pd.Series(mi_scores, index=feature_cols)
            for feature, importance in mi_scores.items():
                if feature not in feature_importances:
                    feature_importances[feature] = []
                feature_importances[feature].append(importance)
                
            # Select top features
            k = max(int(len(feature_cols) * feature_keep_percent), 5)  # Keep at least 5 features
            selector = SelectKBest(mutual_info_classif, k=k)
            selector.fit(X_numeric, y)
            mi_selected_features = X_numeric.columns[selector.get_support()].tolist()
            
            # Update kept features
            if mi_selected_features:
                kept_features = kept_features.intersection(mi_selected_features)
                logging.info(f"Mutual info selected {len(mi_selected_features)} features")
        except Exception as e:
            logging.warning(f"Error in mutual information analysis: {e}")
    
    # Step 6: Random Forest feature importance
    if use_random_forest and len(feature_cols) > 5:
        try:
            logging.info("Running Random Forest feature importance analysis...")
            X_rf = X.select_dtypes(include=[np.number])  # Only numeric for RF
            if len(X_rf.columns) > 0:
                # Impute missing values before fitting RandomForest
                imputer = SimpleImputer(strategy='mean')
                X_rf_imputed = pd.DataFrame(imputer.fit_transform(X_rf), columns=X_rf.columns, index=X_rf.index)
                # Train a Random Forest with limited depth for speed
                rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                rf.fit(X_rf_imputed, y)
                
                # Get feature importances
                importances = rf.feature_importances_
                rf_importance = pd.Series(importances, index=X_rf.columns)
                for feature, importance in rf_importance.items():
                    if feature not in feature_importances:
                        feature_importances[feature] = []
                    feature_importances[feature].append(importance * 100)  # Scale for readability
                
                # Select top features
                k = max(int(len(X_rf.columns) * feature_keep_percent), 5)
                top_rf_features = rf_importance.nlargest(k).index.tolist()
                
                # Update kept features
                if top_rf_features:
                    # If we have results from mutual_info, intersect; otherwise use RF results
                    if kept_features == set(feature_cols):
                        kept_features = set(top_rf_features)
                    else:
                        kept_features = kept_features.intersection(set(top_rf_features))
                    logging.info(f"Random Forest selected {len(top_rf_features)} features")
        except Exception as e:
            logging.warning(f"Error in Random Forest feature selection: {e}")
    
    # Make sure we keep some features by taking the top N if intersection is too restrictive
    if len(kept_features) < max(3, int(len(feature_cols) * 0.3)):
        # Calculate average importance across methods
        avg_importance = {}
        for feature, scores in feature_importances.items():
            avg_importance[feature] = sum(scores) / len(scores)
        
        # Take top 30% features
        k = max(int(len(feature_cols) * 0.3), 5)
        kept_features = set(sorted(avg_importance.keys(), key=lambda x: avg_importance[x], reverse=True)[:k])
        logging.info(f"Using top {len(kept_features)} features by average importance")
    
    # Generate feature importance report
    if generate_report:
        try:
            report_path = f"feature_importance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            logging.info(f"Generating feature importance report: {report_path}")
            
            # Average importance scores across methods
            avg_importance = {}
            for feature, scores in feature_importances.items():
                avg_importance[feature] = sum(scores) / len(scores)
                
            # Create sorted importance DataFrame
            importance_df = pd.DataFrame({
                'Feature': list(avg_importance.keys()),
                'Importance': list(avg_importance.values())
            }).sort_values('Importance', ascending=False).head(30)  # Top 30 features
            
            # Plot importance
            plt.figure(figsize=(12, 10))
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig(report_path)
            logging.info(f"Feature importance plot saved to {report_path}")
        except Exception as e:
            logging.warning(f"Could not generate feature importance report: {e}")
    
    # Get final columns to keep
    columns_to_keep = list(kept_features) + non_feature_cols
    df_optimized = df[columns_to_keep]
    
    # Verify we still have the label column
    if label_col not in df_optimized.columns:
        logging.warning(f"Label column '{label_col}' missing after optimization, adding back.")
        df_optimized[label_col] = df[label_col]
    
    # Final log
    logging.info(f"Data optimization complete: {original_shape[0]} rows, {original_shape[1]} columns -> "
                f"{df_optimized.shape[0]} rows, {df_optimized.shape[1]} columns")
    logging.info(f"Final columns: {list(df_optimized.columns)}")
    
    return df_optimized

def augment_with_gaussian_noise(df, noise_std=0.01, noise_prob=0.5, exclude_cols=None):
    """
    Augment numeric columns by injecting small Gaussian noise to a random subset of rows.
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
    parser = argparse.ArgumentParser(description="Optimize tabular data for AI model training.")
    parser.add_argument('--input', type=str, required=True, help="Input CSV file to optimize")
    parser.add_argument('--output', type=str, required=True, help="Output CSV file for optimized data")
    parser.add_argument('--label-col', type=str, default='label', help="Label column name")
    parser.add_argument('--text-col', type=str, default='text', help="Text column name")
    parser.add_argument('--corr-thresh', type=float, default=0.95, help="Correlation threshold (0.0-1.0)")
    parser.add_argument('--feature-keep', type=float, default=0.7, help="Percentage of features to keep (0.0-1.0)")
    parser.add_argument('--disable-mutual-info', action='store_true', help="Disable mutual information selection")
    parser.add_argument('--disable-random-forest', action='store_true', help="Disable Random Forest selection")
    parser.add_argument('--disable-report', action='store_true', help="Disable feature importance report")
    args = parser.parse_args()

    # SageMaker: Use environment variables if present
    input_path = os.environ.get('SM_CHANNEL_TRAIN', args.input)
    output_path = os.environ.get('SM_OUTPUT_DATA_DIR', args.output)

    if not os.path.isfile(input_path):
        logging.error(f"Input file not found: {input_path}")
        exit(1)

    logging.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    df_optimized = optimize_data_for_ai(
        df, 
        label_col=args.label_col, 
        text_col=args.text_col,
        corr_thresh=args.corr_thresh,
        feature_keep_percent=args.feature_keep,
        use_mutual_info=not args.disable_mutual_info,
        use_random_forest=not args.disable_random_forest,
        generate_report=not args.disable_report
    )
    
    # Define non_feature_cols for augmentation
    non_feature_cols = [args.label_col, args.text_col]
    df = augment_with_gaussian_noise(df, noise_std=0.01, noise_prob=0.5, exclude_cols=non_feature_cols)
    logging.info("Applied Gaussian noise augmentation to numeric features.")
    
    df_optimized.to_csv(output_path, index=False)
    logging.info(f"Optimized data saved to {output_path}")

if __name__ == "__main__":
    main()
