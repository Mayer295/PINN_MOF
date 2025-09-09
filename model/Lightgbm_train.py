import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import optuna
from tqdm import tqdm
import matplotlib

# Set matplotlib backend to TkAgg, suitable for Tkinter environment
matplotlib.use('TkAgg')

# Load data
file_path = r'E:\mof训练-PINN\mof数据\ps_usable_hydrogen_storage_capacity_gcmcv2.xlsx'
df = pd.read_excel(file_path)
# Remove spaces from feature names
df.columns = df.columns.str.replace(' ', '_')

# Select features and target variable
X = df.iloc[:, 5:12]  # Columns 6 to 12 as input features
y = df.iloc[:, 12]    # Column 13 as output (gravimetric capacity, wt.%)

# Check for missing values or outliers in data
print("Check for missing values:", df.isnull().sum())
print("Descriptive statistics of target variable:", y.describe())

# Split data into training (70%), validation (15%), and test (15%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define evaluation function
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    std_actual = np.std(y_true)
    std_predicted = np.std(y_pred)
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    return rmse, mae, r2, std_actual, std_predicted, correlation

# Model file path - specify full path directly
model_path = r'E:\mof训练-PINN\LightGBM\ug\lgb_ug_train.pkl'

# Ensure save directory exists
model_dir = os.path.dirname(model_path)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"Created directory: {model_dir}")

# Check if model exists
if os.path.exists(model_path):
    # Load existing model
    best_model = joblib.load(model_path)
    print("Model loaded from disk.")
else:
    # Define Optuna objective function
    def objective(trial):
        param_grid = {
            'objective': 'regression',  # Regression task
            'metric': 'rmse',           # Evaluation metric is RMSE
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),           # Number of leaves
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3), # Learning rate
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),     # Number of trees
            'max_depth': trial.suggest_int('max_depth', -1, 50),              # Maximum depth
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),  # Minimum child samples
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),         # Subsample ratio
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),  # Feature sampling ratio
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),         # L1 regularization
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)        # L2 regularization
        }

        # Create LightGBM datasets
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        # Train model with early stopping
        model = lgb.train(param_grid, dtrain, valid_sets=[dval],
                          callbacks=[lgb.early_stopping(stopping_rounds=10)])

        # Predict on validation set
        y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
        return rmse_val

    # Run Optuna optimization with progress bar
    class TqdmCallback:
        def __init__(self, total):
            self.pbar = tqdm(total=total)

        def __call__(self, study, trial):
            self.pbar.update(1)

    # Create Optuna study
    study = optuna.create_study(direction='minimize')  # Goal is to minimize RMSE
    tqdm_callback = TqdmCallback(total=100)
    study.optimize(objective, n_trials=100, callbacks=[tqdm_callback])

    # Output best parameters
    best_params = study.best_params
    print(f"Best parameters found: {best_params}")

    # Clean parameter dictionary to avoid warnings
    best_params.pop('n_estimators', None)

    # Train final model with best parameters
    best_model = lgb.LGBMRegressor(**best_params, n_estimators=study.best_trial.params['n_estimators'])
    best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse',
                   callbacks=[lgb.log_evaluation(period=0)])

    # Save trained model
    joblib.dump(best_model, model_path)
    print("Model training completed and saved.")

# Predict on training, validation, and test sets
y_train_pred = best_model.predict(X_train)
y_val_pred = best_model.predict(X_val)
y_test_pred = best_model.predict(X_test)

# Evaluate performance on training, validation, and test sets
rmse_train, mae_train, r2_train, std_train_actual, std_train_predicted, corr_train = evaluate(y_train, y_train_pred)
rmse_val, mae_val, r2_val, std_val_actual, std_val_predicted, corr_val = evaluate(y_val, y_val_pred)
rmse_test, mae_test, r2_test, std_test_actual, std_test_predicted, corr_test = evaluate(y_test, y_test_pred)

# Print evaluation metrics
print(f"Training set evaluation - RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}, R2: {r2_train:.4f}, Std actual: {std_train_actual:.4f}, Std predicted: {std_train_predicted:.4f}, Correlation: {corr_train:.4f}")
print(f"Validation set evaluation - RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}, R2: {r2_val:.4f}, Std actual: {std_val_actual:.4f}, Std predicted: {std_val_predicted:.4f}, Correlation: {corr_val:.4f}")
print(f"Test set evaluation - RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}, R2: {r2_test:.4f}, Std actual: {std_test_actual:.4f}, Std predicted: {std_test_predicted:.4f}, Correlation: {corr_test:.4f}")

# Plotting function
def plot_comparison(y_true, y_pred, set_name, rmse, mae, r2, save_path=None):
    # Format evaluation metrics to control precision
    def format_metric(value):
        return f'{value:.4f}'[:f'{value:.4f}'.find('.') + 3]

    mae_str = format_metric(mae)
    rmse_str = format_metric(rmse)
    r2_str = format_metric(r2)

    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, y_pred, color='blue', alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    plt.xlabel('GCMC calculated gravimetric capacity (wt. %)', fontsize=10)
    plt.ylabel('ML predicted gravimetric capacity (wt. %)', fontsize=10)
    plt.title(f'{set_name} Set Prediction Comparison')

    # Display evaluation metrics on the plot
    plt.text(min(y_true), max(y_pred) * 0.95,  # Adjust text position to avoid overflow
             f'R$^{{2}}$: {r2_str}\nMAE: {mae_str}\nRMSE: {rmse_str}',
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.grid(False)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    plt.show()

# Plot and save comparison for test set using direct path
save_path = r'E:\mof训练-PINN\LightGBM\ug\light_gbm_ug_comparison_test.png'
plot_comparison(y_test, y_test_pred, "Test", rmse_test, mae_test, r2_test, save_path=save_path)