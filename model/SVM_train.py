import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import joblib
import matplotlib
from tqdm import tqdm
import optuna
from optuna.pruners import MedianPruner

# Set matplotlib backend to TkAgg, suitable for Tkinter
matplotlib.use('TkAgg')

# Load Excel file
print("Loading data...")
with tqdm(total=1, desc="Loading data") as pbar:
    file_path = r'ps_usable_hydrogen_storage_capacity_gcmcv2.xlsx'
    df = pd.read_excel(file_path)
    pbar.update(1)

# Extract input features and target variable
X = df.iloc[:, 5:12]  # Columns 6 to 12 as input features
y = df.iloc[:, 12]    # Column 13 as target variable

# Split dataset into training, validation, and test sets
print("Splitting dataset...")
with tqdm(total=2, desc="Splitting dataset") as pbar:
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    pbar.update(1)
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(X_temp_raw, y_temp, test_size=0.5, random_state=42)
    pbar.update(1)

# Model save paths
model_path = r'best_svr_ug_train.pkl'
scaler_path = r'scaler_svr_ug.pkl'

# Ensure save directory exists
model_dir = os.path.dirname(model_path)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"Created directory: {model_dir}")

# Check if model and scaler already exist
if os.path.exists(model_path) and os.path.exists(scaler_path):
    print("Loading model and scaler...")
    with tqdm(total=2, desc="Loading model") as pbar:
        best_svr = joblib.load(model_path)
        pbar.update(1)
        scaler = joblib.load(scaler_path)
        pbar.update(1)
    print(f"Model and scaler loaded from {model_path}")
else:
    # Define optimized Optuna objective function
    def objective(trial):
        # Optimized parameter search space
        C = trial.suggest_float('C', 0.05, 30, log=True)
        gamma_option = trial.suggest_categorical('gamma_option', ['scale', 'auto', 'value'])
        if gamma_option == 'value':
            gamma = trial.suggest_float('gamma_value', 0.001, 0.3, log=True)
        else:
            gamma = gamma_option
        epsilon = trial.suggest_float('epsilon', 0.01, 0.2)

        # Add tol parameter to search space to address convergence issues
        tol = trial.suggest_float('tol', 1e-5, 1e-3, log=True)

        # Create and train model
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('svr', SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon,
                      max_iter=100000, tol=tol, cache_size=2000))  # Increase cache_size for computational efficiency
        ])

        # Evaluate directly on the full validation set
        pipeline.fit(X_train_raw, y_train)
        y_val_pred = pipeline.predict(X_val_raw)

        # Return negative MSE
        return -mean_squared_error(y_val, y_val_pred)

    print("Starting Optuna hyperparameter optimization...")
    # Use MedianPruner to terminate poorly performing trials early
    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
    )

    # Use parallel processing to speed up Optuna optimization
    with tqdm(total=1, desc="Optuna optimization") as pbar:
        study.optimize(objective, n_trials=20, n_jobs=-1, show_progress_bar=True)  # Add n_jobs=-1 for parallel processing
        pbar.update(1)

    # Print best parameters
    print(f"Best parameters: {study.best_params}")
    print(f"Best value (neg_mean_squared_error): {study.best_value}")

    # Create final model with best parameters
    best_params = study.best_params

    # Handle gamma parameter
    if best_params['gamma_option'] == 'value':
        best_gamma = best_params['gamma_value']
    else:
        best_gamma = best_params['gamma_option']

    # Create final model
    best_pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('svr', SVR(kernel='rbf', C=best_params['C'], gamma=best_gamma,
                 epsilon=best_params['epsilon'], max_iter=100000,
                 tol=best_params['tol'], cache_size=2000))  # Use optimized tol value
    ])

    # Train final model
    print("Training final model...")
    best_pipeline.fit(X_train_raw, y_train)
    best_svr = best_pipeline

    # Save model and scaler
    print("Saving model and scaler...")
    with tqdm(total=2, desc="Saving model") as pbar:
        joblib.dump(best_svr, model_path)
        pbar.update(1)
        scaler = best_svr.named_steps['scaler']
        joblib.dump(scaler, scaler_path)
        pbar.update(1)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

# Make predictions
print("Making predictions...")
with tqdm(total=3, desc="Predicting") as pbar:
    y_train_pred = best_svr.predict(X_train_raw)
    pbar.update(1)
    y_val_pred = best_svr.predict(X_val_raw)
    pbar.update(1)
    y_test_pred = best_svr.predict(X_test_raw)
    pbar.update(1)

# Calculate evaluation metrics
print("Calculating evaluation metrics...")
with tqdm(total=9, desc="Calculating metrics") as pbar:
    mae_train = mean_absolute_error(y_train, y_train_pred)
    pbar.update(1)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    pbar.update(1)
    r2_train = r2_score(y_train, y_train_pred)
    pbar.update(1)

    mae_val = mean_absolute_error(y_val, y_val_pred)
    pbar.update(1)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
    pbar.update(1)
    r2_val = r2_score(y_val, y_val_pred)
    pbar.update(1)

    mae_test = mean_absolute_error(y_test, y_test_pred)
    pbar.update(1)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    pbar.update(1)
    r2_test = r2_score(y_test, y_test_pred)
    pbar.update(1)

# Print evaluation metrics
print("Training set evaluation metrics:")
print(f"MAE: {mae_train:.4f}")
print(f"RMSE: {rmse_train:.4f}")
print(f"R2: {r2_train:.4f}")

print("Validation set evaluation metrics:")
print(f"MAE: {mae_val:.4f}")
print(f"RMSE: {rmse_val:.4f}")
print(f"R2: {r2_val:.4f}")

print("Test set evaluation metrics:")
print(f"MAE: {mae_test:.4f}")
print(f"RMSE: {rmse_test:.4f}")
print(f"R2: {r2_test:.4f}")

# Plotting function remains unchanged
def plot_comparison(y_true, y_pred, set_name, r2, mae, rmse, save_path=None):
    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, y_pred, color='blue', alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    plt.xlabel('GCMC calculated gravimetric capacity (wt. %)', fontsize=10)
    plt.ylabel('ML predicted gravimetric capacity (wt. %)', fontsize=10)

    # Display metrics, truncated to two decimal places
    r2_str = str(r2)[:str(r2).find('.') + 3]
    mae_str = str(mae)[:str(mae).find('.') + 3]
    rmse_str = str(rmse)[:str(rmse).find('.') + 3]

    plt.text(min(y_true), max(y_pred),
             f'R$^{2}$: {r2_str}\nMAE: {mae_str}\nRMSE: {rmse_str}',
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.title(f'{set_name} Set Prediction Comparison')
    plt.grid(False)
    if save_path:
        with tqdm(total=1, desc="Saving plot") as pbar:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            pbar.update(1)
        print(f"Saved plot to: {save_path}")
    plt.show()

# Generate and save comparison plot
print("Generating comparison plot...")
plot_comparison(y_test, y_test_pred, 'Test', r2_test, mae_test, rmse_test,
                save_path=r'svr_comparison_ug.png')