import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import optuna
import matplotlib

matplotlib.use('TkAgg')  # Set matplotlib backend to TkAgg


def load_and_preprocess_data(file_path):
    """Load and preprocess data"""
    try:
        data = pd.read_excel(file_path)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Failed to load data: {e}")
        sys.exit(1)

    # Data preprocessing
    X = data.iloc[:, 5:12].values  # Columns 6-12 as input features
    y = data.iloc[:, 12].values  # Column 13 as output feature

    # Split dataset into training and temporary sets (85% training, 15% temporary)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.15, random_state=42)

    # Split temporary set into validation and test sets (50% validation, 50% test)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Standardize input features only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Save the scaler
    scaler_path = r'scaler_nn_ps_ug.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    print(f"Training set input shape: {X_train.shape}, output shape: {y_train.shape}")
    print(f"Validation set input shape: {X_val.shape}, output shape: {y_val.shape}")
    print(f"Test set input shape: {X_test.shape}, output shape: {y_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def create_model(trial, input_shape):
    """Create model with hyperparameters suggested by Optuna"""
    # Define hyperparameters to optimize
    n_layers = trial.suggest_int('n_layers', 2, 5)
    first_units = trial.suggest_categorical('first_units', [64, 96, 128, 256])
    activation = trial.suggest_categorical('activation', ['relu', 'elu', 'selu'])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

    # Build the model
    model = Sequential()
    model.add(Dense(first_units, activation=activation, input_shape=input_shape))

    # Add hidden layers
    for i in range(n_layers - 1):
        # Number of units decreases with each layer
        units = trial.suggest_int(f'units_l{i}',
                                  max(8, first_units // (2 ** (i + 1))),
                                  first_units // (2 ** i))
        model.add(Dense(units, activation=activation))

    # Output layer
    model.add(Dense(1))

    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['mae']
    )

    return model


def objective(trial, X_train, X_val, y_train, y_val, input_shape):
    """Objective function for Optuna optimization"""
    # Create model with trial hyperparameters
    model = create_model(trial, input_shape)

    # Define batch size and early stopping
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    patience = trial.suggest_int('patience', 20, 50)

    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )

    # Train the model
    model.fit(
        X_train, y_train,
        epochs=300,  # Maximum number of epochs
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=0  # Silent training
    )

    # Evaluate on validation set
    val_loss = model.evaluate(X_val, y_val, verbose=0)[0]

    return val_loss


def get_best_model(study, X_train, X_val, y_train, y_val, input_shape):
    """Get the best model based on Optuna results"""
    print("Training the best model with optimized hyperparameters...")
    best_params = study.best_params

    # Create model with best parameters
    model = Sequential()
    model.add(Dense(best_params['first_units'],
                    activation=best_params['activation'],
                    input_shape=input_shape))

    # Add hidden layers
    for i in range(best_params['n_layers'] - 1):
        if f'units_l{i}' in best_params:
            model.add(Dense(best_params[f'units_l{i}'],
                            activation=best_params['activation']))

    # Output layer
    model.add(Dense(1))

    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['mae']
    )

    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=best_params['patience'],
        restore_best_weights=True,
        verbose=1
    )

    # Train the model with best hyperparameters
    model.fit(
        X_train, y_train,
        epochs=300,
        batch_size=best_params['batch_size'],
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )

    return model


def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test):
    """Evaluate model performance and calculate metrics"""
    try:
        y_train_pred = model.predict(X_train, verbose=0)
        y_val_pred = model.predict(X_val, verbose=0)
        y_test_pred = model.predict(X_test, verbose=0)
        print(
            f"Prediction shapes: Training {y_train_pred.shape}, Validation {y_val_pred.shape}, Test {y_test_pred.shape}")
    except Exception as e:
        print(f'Error predicting data: {e}')
        print(f"Expected input shape: {model.input_shape}")
        sys.exit(1)

    # Calculate evaluation metrics
    metrics = {}
    metrics['train'] = {
        'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred.flatten())),
        'mae': mean_absolute_error(y_train, y_train_pred.flatten()),
        'r2': r2_score(y_train, y_train_pred.flatten())
    }
    metrics['val'] = {
        'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred.flatten())),
        'mae': mean_absolute_error(y_val, y_val_pred.flatten()),
        'r2': r2_score(y_val, y_val_pred.flatten())
    }
    metrics['test'] = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred.flatten())),
        'mae': mean_absolute_error(y_test, y_test_pred.flatten()),
        'r2': r2_score(y_test, y_test_pred.flatten())
    }

    # Print evaluation results
    print(f'Training set evaluation results:')
    print(f'RMSE={metrics["train"]["rmse"]:.4f}, MAE={metrics["train"]["mae"]:.4f}, R^2={metrics["train"]["r2"]:.4f}')
    print(f'Validation set evaluation results:')
    print(f'RMSE={metrics["val"]["rmse"]:.4f}, MAE={metrics["val"]["mae"]:.4f}, R^2={metrics["val"]["r2"]:.4f}')
    print(f'Test set evaluation results:')
    print(f'RMSE={metrics["test"]["rmse"]:.4f}, MAE={metrics["test"]["mae"]:.4f}, R^2={metrics["test"]["r2"]:.4f}')

    return y_test_pred, metrics


def plot_results(y_test, y_test_pred, metrics, save_path):
    """Plot comparison of predicted vs actual values"""
    test_r2_str = f"{metrics['test']['r2']:.4f}"
    test_mae_str = f"{metrics['test']['mae']:.4f}"
    test_rmse_str = f"{metrics['test']['rmse']:.4f}"

    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_test_pred.flatten(), color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('GCMC calculated gravimetric capacity (wt. %)', fontsize=10)
    plt.ylabel('ML predicted gravimetric capacity (wt. %)', fontsize=10)

    plt.text(min(y_test), max(y_test_pred.flatten()),
             f'R$^{{2}}$: {test_r2_str}\nMAE: {test_mae_str}\nRMSE: {test_rmse_str}',
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.grid(False)
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    print(f"TensorFlow version: {tf.__version__}")

    # File paths configuration
    data_file = r'ps_usable_hydrogen_storage_capacity_gcmcv2.xlsx'
    model_file = r'nn_ps_ug_optuna.h5'
    result_path = r'nn_ps_ug_optuna_pred_test.png'
    optuna_db_path = r'optuna_ps_ug.db'

    # Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_and_preprocess_data(data_file)
    input_shape = (X_train.shape[1],)

    # Check if model exists
    if os.path.exists(model_file) and not input("Model exists. Retrain? (y/n): ").lower().startswith('y'):
        try:
            model = load_model(model_file)
            print(f'Model loaded successfully from {model_file}')
            model.summary()
        except Exception as e:
            print(f'Failed to load model: {e}')
            print("Will create and train a new model...")
            model = None
    else:
        model = None
        print("Will create and train a new model...")

    # If no model is loaded, perform Optuna optimization
    if model is None:
        print("Starting Optuna hyperparameter optimization...")

        # Create a study object
        storage = f"sqlite:///{optuna_db_path}"
        study = optuna.create_study(
            direction="minimize",
            study_name="nn_ps_ug_optimization",
            storage=storage,
            load_if_exists=True
        )

        # Run optimization
        study.optimize(
            lambda trial: objective(trial, X_train, X_val, y_train, y_val, input_shape),
            n_trials=30,  # Number of trials to run
            timeout=3600  # Maximum time in seconds (1 hour)
        )

        # Print optimization results
        print("Optimization completed.")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best value (validation loss): {study.best_value:.6f}")
        print("Best hyperparameters:")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")

        # Train model with best parameters
        model = get_best_model(study, X_train, X_val, y_train, y_val, input_shape)

        # Save the optimized model
        model.save(model_file)
        print(f'Model trained and saved to {model_file}')

    # Evaluate model and get test set predictions
    y_test_pred, metrics = evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test)

    # Plot results
    plot_results(y_test, y_test_pred, metrics, result_path)
    print(f"Results plot saved to {result_path}")


if __name__ == "__main__":
    main()