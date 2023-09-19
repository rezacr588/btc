from functions import preprocess_data, feature_engineering, split_data, create_lstm_model, get_or_create_model

def train_model(url, epochs=50, batch_size=32):
    """
    Train an LSTM model on the data from the provided URL.
    
    Parameters:
    - url: The URL to fetch the data from.
    - epochs: Number of epochs for training.
    - batch_size: Batch size for training.
    
    Returns:
    - model: The trained LSTM model.
    - history: Training history.
    """
    
    # Step 1: Data Preprocessing
    raw_data = preprocess_data(url)
    
    # Step 2: Feature Engineering
    engineered_data = feature_engineering(raw_data)
    
    # Step 3: Splitting the Data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(engineered_data)
    # Step 4: Model Creation
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = get_or_create_model(input_shape)
    
    # Step 5: Model Training
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
    
    # Step 6: Saving the Model
    model.save("output/models/btc_model.h5")
    
    # Step 7: Evaluate the Model on the Test Set
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}")
    print(f"Test MAE: {test_mae}")


    
    return model, history

# Usage
url = 'https://bitcoin-data-collective.vercel.app/api/download_btc'

model, history = train_model(url)
print("Model trained successfully!")
