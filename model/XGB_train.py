import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import optuna
from tqdm import tqdm

matplotlib.use('TkAgg')  # Set matplotlib backend to TkAgg, suitable for Tkinter

# Load Excel file
file_path = r'ps_usable_hydrogen_storage_capacity_gcmcv2.xlsx'
df = pd.read_excel(file_path)

# Extract input features and target variable
X = df.iloc[:, 5:12]  # Columns 6 to 12 as input features
y = df.iloc[:, 12]    # Column 13 as target variable

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Model save path
model_path = r'xgboost_train_ug.pkl'

# Check if model exists
if os.path.exists(model_path):
    # Load existing model
    model = joblib.load(model_path)
    print("Loaded existing model.")
else:
    # Define Optuna objective function
    def objective(trial):
        param_grid = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),         # Number of trees
            'max_depth': trial.suggest_int('max_depth', 3, 10),                   # Maximum depth
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),  # Learning rate
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),             # Subsample ratio
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),  # Feature sampling ratio
            'gamma': trial.suggest_float('gamma', 0, 0.5),                       # Minimum loss reduction
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),    # Minimum child weight
            'objective': 'reg:squarederror',                                      # Regression task objective
            'tree_method': 'hist',                                                # Use hist algorithm
            'device': 'cuda'                                                      # Use CUDA device
        }

        # Initialize XGBoost model
        model = xgb.XGBRegressor(**param_grid)
        model.fit(X_train, y_train)

        # Predict on validation set and calculate MSE
        y_val_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_val_pred)
        return mse

    # Create Optuna study and display progress bar
    study = optuna.create_study(direction='minimize')  # Goal is to minimize MSE
    with tqdm(total=50, desc="Optuna optimization progress", ncols=100) as pbar:
        def callback(study, trial):
            pbar.update(1)
        study.optimize(objective, n_trials=50, callbacks=[callback])

    # Get best parameters
    best_params = study.best_params
    print(f"Best parameters: {best_params}")
    print(f"Best validation MSE: {study.best_value}")

    # Train final model with best parameters (using GPU)
    best_params['objective'] = 'reg:squarederror'
    best_params['tree_method'] = 'hist'
    best_params['device'] = 'cuda'
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, model_path)
    print("New model trained and saved.")

# Predict on training, validation, and test sets
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Define evaluation metrics function
def print_metrics(y_true, y_pred, dataset_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    std_actual = np.std(y_true)
    std_predicted = np.std(y_pred)
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    print(f"{dataset_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, "
          f"Std actual: {std_actual:.4f}, Std predicted: {std_predicted:.4f}, "
          f"Correlation: {correlation:.4f}")

# Print evaluation metrics for training, validation, and test sets
print_metrics(y_train, y_train_pred, 'Training Set')
print_metrics(y_val, y_val_pred, 'Validation Set')
print_metrics(y_test, y_test_pred, 'Test Set')

# Define plotting function
def plot_comparison(y_true, y_pred, save_path=None):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Format evaluation metrics to control precision without rounding
    def format_metric(value):
        return f'{value:.4f}'[:f'{value:.4f}'.find('.') + 3]

    mae_str = format_metric(mae)
    rmse_str = format_metric(rmse)
    r2_str = format_metric(r2)

    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, y_pred, color='blue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    plt.xlabel('GCMC calculated gravimetric capacity (wt. %)', fontsize=10)
    plt.ylabel('ML predicted gravimetric capacity (wt. %)', fontsize=10)

    # Display evaluation metrics on the plot for test set
    plt.text(min(y_true), max(y_pred),
             f' R$^{2}$: {r2_str}\n MAE: {mae_str}\n RMSE: {rmse_str}',
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.grid(False)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

# Call plotting function and save image
plot_comparison(y_test, y_test_pred, save_path='ps_ug_pred_xgboost_test.png')