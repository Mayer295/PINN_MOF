import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import optuna

matplotlib.use('TkAgg')  # Set matplotlib backend to TkAgg, suitable for GUI-supporting environments

# Read Excel file
data = pd.read_excel(r'\ps_usable_hydrogen_storage_capacity_gcmcv2.xlsx')

# Select input features and target variable
X = data.iloc[:, 5:12]  # Columns 6 to 12 (zero-based indexing)
y = data.iloc[:, 12]  # Column 13 (zero-based indexing)

# Split data into training (70%), validation (15%), and test (15%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)  # First split: 70% train, 30% temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5,
                                                random_state=42)  # Split temp into 15% val and 15% test

# Model file path
model_filename = r'\random_forest_train_ug_compressed.joblib'

# Check if directory exists, create it if it doesn't
model_dir = os.path.dirname(model_filename)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


# Define Optuna objective function
def objective(trial):
    # Define hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 50, log=True),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }

    # Initialize Random Forest model
    model = RandomForestRegressor(**params, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

    return val_rmse


# Check if model exists
if os.path.exists(model_filename):
    try:
        # If model exists, load it
        model = joblib.load(model_filename)
        print("Model already exists, loading the model...")

        # Predict on training, validation, and test sets
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        # Calculate evaluation metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)

        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_r2 = r2_score(y_val, y_val_pred)

        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)

        # Print evaluation metrics
        print("Training set evaluation metrics (loaded model):")
        print(f"MAE: {train_mae:.4f}")
        print(f"RMSE: {train_rmse:.4f}")
        print(f"R2: {train_r2:.4f}")

        print("Validation set evaluation metrics (loaded model):")
        print(f"MAE: {val_mae:.4f}")
        print(f"RMSE: {val_rmse:.4f}")
        print(f"R2: {val_r2:.4f}")

        print("Test set evaluation metrics (loaded model):")
        print(f"MAE: {test_mae:.4f}")
        print(f"RMSE: {test_rmse:.4f}")
        print(f"R2: {test_r2:.4f}")

    except Exception as e:
        print(f"Error loading the model: {e}. Retraining the model with Optuna...")
        model = None
else:
    print("Model does not exist, training a new model with Optuna...")
    model = None

# If model doesn't exist or loading fails, optimize with Optuna and train a new model
if model is None:
    # Create Optuna study
    study = optuna.create_study(direction='minimize')  # Goal is to minimize validation RMSE
    study.optimize(objective, n_trials=50)  # Run 50 trials

    # Get best parameters
    best_params = study.best_params
    print("Best parameters found by Optuna:", best_params)

    # Train final model with best parameters
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)

    # Save model with compression
    joblib.dump(model, model_filename, compress=3)
    print("Model training completed and saved...")

    # Predict on training, validation, and test sets
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Calculate evaluation metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)

    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    # Print evaluation metrics
    print("Training set evaluation metrics (new model):")
    print(f"MAE: {train_mae:.4f}")
    print(f"RMSE: {train_rmse:.4f}")
    print(f"R2: {train_r2:.4f}")

    print("Validation set evaluation metrics (new model):")
    print(f"MAE: {val_mae:.4f}")
    print(f"RMSE: {val_rmse:.4f}")
    print(f"R2: {val_r2:.4f}")

    print("Test set evaluation metrics (new model):")
    print(f"MAE: {test_mae:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"R2: {test_r2:.4f}")


# Format evaluation metrics to control precision without rounding
def format_metric(value):
    return f'{value:.4f}'[:f'{value:.4f}'.find('.') + 3]


test_mae_str = format_metric(test_mae)
test_rmse_str = format_metric(test_rmse)
test_r2_str = format_metric(test_r2)

# Plot comparison graph for test set and save it with evaluation metrics
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_test_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('GCMC calculated gravimetric capacity (wt. %)', fontsize=10)
plt.ylabel('ML predicted gravimetric capacity (wt. %)', fontsize=10)

# Display test set evaluation metrics on the plot
plt.text(min(y_test), max(y_test_pred),
         f'R$^{2}$: {test_r2_str}\nMAE: {test_mae_str}\nRMSE: {test_rmse_str}',
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.grid(False)
plt.savefig('RF_ps_ug_pred_test.png', dpi=300)
plt.show()