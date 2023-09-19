import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def time_based_split(data, test_size=0.2):
    """
    Splits the data into training and testing sets based on the given test size.
    
    Parameters:
    - data (DataFrame): The dataset to be split.
    - test_size (float): Proportion of the dataset to include in the test split (e.g., 0.2 for 20%).
    
    Returns:
    - DataFrame, DataFrame: Training and testing datasets.
    """
    # Calculate the index at which to split the data
    split_idx = int(len(data) * (1 - test_size))
    
    # Split the data
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    return train_data, test_data

def prepare_data(train_data, test_data):
    """Separate features and target from training and testing data."""
    X_train = train_data.drop(columns=['LAST_PRICE', 'TIME'])
    y_train = train_data['LAST_PRICE']

    X_test = test_data.drop(columns=['LAST_PRICE', 'TIME'])
    y_test = test_data['LAST_PRICE']
    
    return X_train, y_train, X_test, y_test

def train_xgboost(X_train, y_train, params=None, num_rounds=100):
    """Train an XGBoost model and save it."""
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'verbosity': 1
        }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    bst = xgb.train(params, dtrain, num_rounds)
    
    # Save the trained model
    bst.save_model('output/models/xgboost_model.model')

    return bst

def load_xgboost_model():
    """Load a pre-trained XGBoost model."""
    bst = xgb.Booster()  # Initialize model
    bst.load_model('output/models/xgboost_model.model')  # Load the model
    return bst

def evaluate_model(bst, X_test, y_test):
    """Evaluate the XGBoost model and return RMSE."""
    dtest = xgb.DMatrix(X_test)
    y_pred = bst.predict(dtest)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    return rmse

def plot_feature_importance(bst):
    """Plot feature importance for the XGBoost model."""
    xgb.plot_importance(bst)
    plt.show()

def continue_training(bst, X_train, y_train, params=None, additional_rounds=100):
    """
    Continue training a pre-loaded XGBoost model.
    
    Parameters:
    - bst (Booster): Pre-trained XGBoost model.
    - X_train (DataFrame): Training features.
    - y_train (Series or array-like): Training target.
    - params (dict, optional): Parameters for XGBoost training. If None, default parameters are used.
    - additional_rounds (int): Number of additional boosting rounds.
    
    Returns:
    - Booster: Updated XGBoost model.
    """
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'verbosity': 1
        }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    bst = xgb.train(params, dtrain, additional_rounds, xgb_model=bst)  # Use xgb_model parameter to continue training
    
    return bst
