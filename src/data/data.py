import os
import requests
import pandas as pd

save_raw_path = 'data/raw/bitcoin_data.csv'

def download_bitcoin_data(save_path=save_raw_path):
    """
    Downloads Bitcoin data from a specified URL and saves it to a given path.
    
    Parameters:
    - save_path (str): Path where the CSV data should be saved. Default is 'data/bitcoin_data.csv'.
    
    Returns:
    - str: Path where the data was saved.
    """
    # URL of the CSV data
    url = "https://bitcoin-data-collective.vercel.app/api/download_btc"

    # Make a request to the URL
    response = requests.get(url)

    # Ensure the request was successful
    response.raise_for_status()

    # Create directory if it doesn't exist
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Write the content to a CSV file
    with open(save_path, 'wb') as file:
        file.write(response.content)

    print(f"Data saved to {save_path}")
    return save_path

def preprocess_bitcoin_data(file_path=save_raw_path):
    """
    Preprocesses the Bitcoin data.
    
    Parameters:
    - file_path (str): Path to the CSV data file.
    
    Returns:
    - DataFrame: Preprocessed data.
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Convert the 'TIME' column to datetime format
    df['TIME'] = pd.to_datetime(df['TIME'], unit='s')
    
    # Check if the data is ordered chronologically
    if not df['TIME'].is_monotonic_increasing:
        df.sort_values(by='TIME', inplace=True)
        print("Data was not ordered chronologically. It has been sorted.")
    else:
        print("Data is ordered chronologically.")
    
    # Handle missing values (for this example, we'll drop them)
    df.dropna(inplace=True)
    
    # Create lag features for 'LAST_PRICE' and 'VOLUME_24H'
    for i in range(1, 61):  # for 60 minutes
        df[f'LAST_PRICE_LAG_{i}'] = df['LAST_PRICE'].shift(i)
        df[f'VOLUME_24H_LAG_{i}'] = df['VOLUME_24H'].shift(i)
    
    # Drop rows with NaN values caused by the lag features
    df.dropna(inplace=True)
    
    return df
