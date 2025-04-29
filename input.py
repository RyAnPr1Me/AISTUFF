import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add the src directory to the Python path
src_path = str(Path(__file__).resolve().parent / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from src.models.stock_ai import MultimodalStockPredictor  # Updated to use MultimodalStockPredictor
except ImportError:
    raise ImportError("The 'MultimodalStockPredictor' class/module could not be found in 'src.models.stock_ai'. Verify the file and class/module name.")

# Set the ticker and model file path
TICKER = 'SMCI'  # Replace with your ticker symbol
MODEL_PATH = 'trained_model/random_forest_model.pkl'

# Load the pre-trained model or create a simple one if it doesn't exist
def load_model(model_path):
    model = joblib.load(model_path)
    return modeljoblib.load(model_path)
        print(f"Loaded existing model from {model_path}")
# Fetch stock data for the past 3 months
def fetch_data(ticker, start_date, end_date):
    print(f"Fetching data for {ticker} from {start_date} to {end_date}")mForest model instead.")
    data = yf.download(ticker, start=start_date, end=end_date)
    print(f"Downloaded {len(data)} rows of data")doesn't exist
    return dataos
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
# Add technical indicators to the stock data
def add_technical_indicators(df):ifier(n_estimators=100, random_state=42)
    # Print column names to debugre use
    print("Available columns:", df.columns.tolist())
        print(f"Created and saved a new model to {model_path}")
    # Check if the DataFrame has a MultiIndex and flatten if needed
    if isinstance(df.columns, pd.MultiIndex):
        # Use the first element (data type) instead of the second (ticker)
        df.columns = [col[0] if isinstance(col, tuple) and len(col) > 1 else col for col in df.columns]
        print("After flattening MultiIndex, columns:", df.columns.tolist())
    data = yf.download(ticker, start=start_date, end=end_date)
    # Find the close price column regardless of capitalization
    close_col = next((col for col in df.columns if col.lower() == 'close'), None)
    if not close_col:
        raise KeyError("Could not find 'Close' column in the DataFrame. Available columns: " + str(df.columns.tolist()))
    add_technical_indicators(df):
    # Moving Averages (MA)o debug
    df['SMA_50'] = df[close_col].rolling(window=50).mean()
    df['SMA_200'] = df[close_col].rolling(window=200).mean()
    # Check if the DataFrame has a MultiIndex and flatten if needed
    # Exponential Moving Average (EMA)Index):
    df['EMA_12'] = df[close_col].ewm(span=12, adjust=False).mean()(ticker)
    df['EMA_26'] = df[close_col].ewm(span=26, adjust=False).mean()) > 1 else col for col in df.columns]
        print("After flattening MultiIndex, columns:", df.columns.tolist())
    # Relative Strength Index (RSI)
    delta = df[close_col].diff(1) regardless of capitalization
    gain = delta.where(delta > 0, 0) df.columns if col.lower() == 'close'), None)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()' column in the DataFrame. Available columns: " + str(df.columns.tolist()))
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))ng(window=50).mean()
    df['SMA_200'] = df[close_col].rolling(window=200).mean()
    # Moving Average Convergence Divergence (MACD)
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['EMA_26'] = df[close_col].ewm(span=26, adjust=False).mean()
    # Bollinger Bands
    df['Bollinger_Upper'] = df['SMA_50'] + (2 * df[close_col].rolling(window=50).std())
    df['Bollinger_Lower'] = df['SMA_50'] - (2 * df[close_col].rolling(window=50).std())
    gain = delta.where(delta > 0, 0)
    # Percentage Price Oscillator (PPO)
    df['PPO'] = ((df['EMA_12'] - df['EMA_26']) / df['EMA_26']) * 100
    avg_loss = loss.rolling(window=14).mean()
    # Drop NaN values (for the initial periods of rolling window calculations)
    df = df.dropna()- (100 / (1 + rs))

    return dfAverage Convergence Divergence (MACD)
    df['MACD'] = df['EMA_12'] - df['EMA_26']
# Preprocess data for model input].ewm(span=9, adjust=False).mean()
def preprocess_data(df):
    # Selecting features
    features = ['SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'Signal_Line', 'Bollinger_Upper', 'Bollinger_Lower', 'PPO']
    df['Bollinger_Lower'] = df['SMA_50'] - (2 * df[close_col].rolling(window=50).std())
    # Check if we have any data
    if df.empty: Price Oscillator (PPO)
        raise ValueError("DataFrame is empty after processing. No data to make predictions.")
    
    # Ensure all required features are present of rolling window calculations)
    df = df[features]
    print(f"Shape after selecting features: {df.shape}")
    return df
    # Standardize the features
    scaler = StandardScaler()nput
    scaled_data = scaler.fit_transform(df)
    # Selecting features
    return scaled_data0', 'SMA_200', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'Signal_Line', 'Bollinger_Upper', 'Bollinger_Lower', 'PPO']
    
# Make prediction using the model
def make_prediction(model, data):
    # Check if the model needs to be trainedis empty after processing. No data to make predictions.")
    if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
        # Model is already trained    # Ensure all required features are present
        prediction = model.predict(data)[features]
    else:
        print("Model has not been trained. Training a simple binary classifier with dummy data.")
        # Create some dummy data for training
        X_dummy = np.random.rand(100, data.shape[1])
        y_dummy = np.random.randint(0, 2, 100)  # Binary classification    scaled_data = scaler.fit_transform(df)
        # Train the model
        model.fit(X_dummy, y_dummy)
        print("Model trained on dummy data. In a real scenario, you should train on actual historical data.")
        # Now make predictiondel
        prediction = model.predict(data)
        prediction = model.predict(data)
    return prediction

def main():def main():
    # Calculate the date range for the past year instead of just 3 monthsdate range for the past year instead of just 3 months
    # to have enough data for 200-day indicatorsday indicators
    end_date = datetime.today()    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)  # Use 365 days instead of 90te - timedelta(days=365)  # Use 365 days instead of 90

    # Fetch the stock data for the past 3 months# Fetch the stock data for the past 3 months
    data = fetch_data(TICKER, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))%m-%d'), end_date.strftime('%Y-%m-%d'))

    # Add technical indicatorstors
    data_with_features = add_technical_indicators(data)ith_features = add_technical_indicators(data)















    main()if __name__ == "__main__":    print(f"Prediction for {TICKER}: {prediction[0]}")        prediction = make_prediction(model, processed_data[-1].reshape(1, -1))  # Use the last entry (most recent data)    # Make a prediction    model = load_model(MODEL_PATH)    # Load the model    processed_data = preprocess_data(data_with_features)    # Preprocess the data    # Preprocess the data
    processed_data = preprocess_data(data_with_features)

    # Load the model
    model = load_model(MODEL_PATH)

    # Make a prediction
    prediction = make_prediction(model, processed_data[-1].reshape(1, -1))  # Use the last entry (most recent data)
    
    print(f"Prediction for {TICKER}: {prediction[0]}")

if __name__ == "__main__":
    main()
