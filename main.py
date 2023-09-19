# main.py

# Import necessary libraries and functions
import os
import pandas as pd
from src.data.data import download_bitcoin_data, preprocess_bitcoin_data
from src.models.model import evaluate_model, plot_feature_importance, prepare_data, time_based_split, train_xgboost, load_xgboost_model, continue_training
import xgboost as xgb
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the data
download_bitcoin_data()
data = preprocess_bitcoin_data()

# Split the data into training and testing sets
train_data, test_data = time_based_split(data, test_size=0.2)

# Prepare data
X_train, y_train, X_test, y_test = prepare_data(train_data, test_data)

# Check if the model is already saved, if not, train a new one
model_path = 'output/models/xgboost_model.model'
if os.path.exists(model_path):
    # Load the pre-trained model
    loaded_model = load_xgboost_model()

    # Continue training the model with additional rounds
    updated_model = continue_training(loaded_model, X_train, y_train, additional_rounds=50)

    # Save the updated model if needed
    updated_model.save_model('output/models/updated_xgboost_model.model')
    bst = updated_model
else:
    bst = train_xgboost(X_train, y_train)

# Convert X_test to DMatrix
dtest = xgb.DMatrix(X_test)

# Predict the price one hour later using the last 60 minutes of test data
last_60_minutes = X_test.iloc[-60:].mean().values.reshape(1, -1)  # Take the mean of the last 60 minutes
d_last_60 = xgb.DMatrix(last_60_minutes)
predicted_price_one_hour_later = bst.predict(d_last_60)
print(f"Predicted price one hour later: {predicted_price_one_hour_later[0]}")

# Evaluate the model
rmse = evaluate_model(bst, X_test, y_test)
print(f"RMSE: {rmse}")