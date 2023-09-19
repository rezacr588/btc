import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import requests
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import os

def preprocess_data(url):
    # Use requests to download the CSV data
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Save the downloaded data to a CSV file
    with open("data/raw/btc.csv", 'wb') as file:
        file.write(response.content)
        
    # Load the data from the CSV URL
    df = pd.read_csv("data/raw/btc.csv")
    
    # Convert the 'Time' column from Unix timestamp to datetime format
    df['TIME'] = pd.to_datetime(df['TIME'], unit='s')
    
    # Sort the dataframe by Time
    df = df.sort_values(by='TIME', ascending=True)
    
    # Handle missing values using the recommended methods
    df.ffill(inplace=True)  # Forward fill
    df.bfill(inplace=True)  # Backward fill
    
    # Scale the numerical columns to be between 0 and 1
    scaler = MinMaxScaler()
    df[['LAST_PRICE', 'ASKS', 'BIDS', 'VOLUME_24H']] = scaler.fit_transform(df[['LAST_PRICE', 'ASKS', 'BIDS', 'VOLUME_24H']])
    
    return df

def feature_engineering(df):
    # Lagged Features for LAST_PRICE (for past 3 minutes)
    for i in range(1, 4):
        df[f'LAST_PRICE_LAG_{i}'] = df['LAST_PRICE'].shift(i)

    # Moving Average for LAST_PRICE (5-minute window)
    df['LAST_PRICE_MA_5'] = df['LAST_PRICE'].rolling(window=5).mean()

    # Price Difference
    df['PRICE_DIFF'] = df['LAST_PRICE'].diff()

    # Extracting hour and day of the week from TIME
    df['HOUR'] = df['TIME'].dt.hour
    df['DAY_OF_WEEK'] = df['TIME'].dt.dayofweek  # Monday=0, Sunday=6

    # Drop rows with NaN values (due to lag and moving average features)
    df.dropna(inplace=True)

    return df

def split_data(df, target_column='LAST_PRICE', val_size=0.2, test_size=0.2, timesteps=1):
    """
    Split the data into training, validation, and test sets and reshape for LSTM.
    
    Parameters:
    - df: The dataframe containing the data.
    - target_column: The column we want to predict.
    - val_size: Proportion of the dataset to include in the validation split.
    - test_size: Proportion of the dataset to include in the test split.
    - timesteps: Number of timesteps for LSTM.
    
    Returns:
    - X_train, X_val, X_test, y_train, y_val, y_test: Training, validation, and test sets.
    """
    
    # Drop the 'TIME' column as it's not a feature for the model
    df = df.drop(columns=['TIME'])
    
    # Splitting the data into features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Splitting into training, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size/(1-test_size), shuffle=False)
    
    # Reshaping the data for LSTM (samples, timesteps, features)
    X_train = X_train.values.reshape((X_train.shape[0], timesteps, X_train.shape[1]))
    X_val = X_val.values.reshape((X_val.shape[0], timesteps, X_val.shape[1]))
    X_test = X_test.values.reshape((X_test.shape[0], timesteps, X_test.shape[1]))
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_lstm_model(input_shape):
    model = Sequential()

    # Input Layer
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    # Additional LSTM Layers
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(50))
    model.add(Dropout(0.2))

    # Dense Layer to produce the forecasted value
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    return model

def get_or_create_model(input_shape, model_path="output/models/btc_model.h5"):
    """
    Load the model if it exists, otherwise create a new one.
    
    Parameters:
    - input_shape: Shape of the input data expected by the LSTM layers.
    - model_path: Path to the saved model file.
    
    Returns:
    - model: Loaded or newly created LSTM model.
    """
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = load_model(model_path)
    else:
        print("Creating new model...")
        model = create_lstm_model(input_shape)
    return model
