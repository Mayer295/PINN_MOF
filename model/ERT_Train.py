import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from joblib import dump
import os
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')  # Set matplotlib backend to TkAgg, suitable for Tkinter

# Load data
df = pd.read_excel(r'\ps_usable_hydrogen_storage_capacity_gcmcv2.xlsx', engine='openpyxl')

# Select columns 6 to 12 as features
X_ug = df.iloc[:, 5:12].values
# Select column 13 as target variable（13:ug  14:uv）
y_ug = df.iloc[:, 12].values

# Split data into training, validation and test sets (ug)
X_train_ug, X_temp_ug, y_train_ug, y_temp_ug = train_test_split(X_ug, y_ug, test_size=0.3, random_state=42)
X_val_ug, X_test_ug, y_val_ug, y_test_ug = train_test_split(X_temp_ug, y_temp_ug, test_size=0.5, random_state=42)

# Initialize Extra Trees regressor
regr = ExtraTreesRegressor(n_estimators=100, random_state=42)

# Train model for ug
regr.fit(X_train_ug, y_train_ug)

# Predict on training, validation and test sets (ug)
y_train_pred_ug = regr.predict(X_train_ug)
y_val_pred_ug = regr.predict(X_val_ug)
y_test_pred_ug = regr.predict(X_test_ug)

# Calculate evaluation metrics for ug
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

print_metrics(y_train_ug, y_train_pred_ug, 'Training Set')
print_metrics(y_val_ug, y_val_pred_ug, 'Validation Set')
print_metrics(y_test_ug, y_test_pred_ug, 'Test Set')

# Plot comparison chart
def plot_comparison(y_true, y_pred, save_path=None):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Format metrics with fixed precision (no rounding)
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

    # Display evaluation metrics on the plot
    plt.text(min(y_true), max(y_pred),
             f'R$^{2}$: {r2_str}\nMAE: {mae_str}\nRMSE: {rmse_str}',
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.grid(False)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

# Call plot function and save image
plot_comparison(y_test_ug, y_test_pred_ug, save_path='ps_ug_pred_extratrees_test.png')

# Save model
model_dir = r'ert_uv_train'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
dump(regr, os.path.join(model_dir, 'ps_ug_train_model.joblib'))