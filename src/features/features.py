
def feature_engineering(df):
    """
    Performs feature engineering on the Bitcoin data.
    
    Parameters:
    - df (DataFrame): The preprocessed Bitcoin data.
    
    Returns:
    - DataFrame: Data with additional engineered features.
    """
    # Lag features for 'LAST_PRICE' and 'VOLUME_24H'
    for i in range(1, 61):  # for 60 minutes
        df[f'LAST_PRICE_LAG_{i}'] = df['LAST_PRICE'].shift(i)
        df[f'VOLUME_24H_LAG_{i}'] = df['VOLUME_24H'].shift(i)
    
    # Rolling statistics
    df['ROLLING_MEAN_LAST_PRICE'] = df['LAST_PRICE'].rolling(window=60).mean()
    df['ROLLING_STD_LAST_PRICE'] = df['LAST_PRICE'].rolling(window=60).std()
    
    # Time-based features
    df['HOUR'] = df['TIME'].dt.hour
    df['DAY_OF_WEEK'] = df['TIME'].dt.dayofweek
    
    # Drop rows with NaN values caused by the lag features and rolling statistics
    df.dropna(inplace=True)
    
    return df