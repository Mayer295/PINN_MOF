import os
import json
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from skopt import BayesSearchCV
from skopt.space import Real
from skopt.callbacks import VerboseCallback
from sklearn.neighbors import KernelDensity
from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib
import math
import joblib  # Add joblib for saving scaler

# Custom KerasRegressor to replace scikeras.wrappers
class CustomKerasRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model=None, lambda_s=0.001, lambda_p=0.1, **kwargs):
        self.model = model
        self.lambda_s = lambda_s
        self.lambda_p = lambda_p
        self.kwargs = kwargs
        self._estimator = None

    def fit(self, X, y, **kwargs):
        if self._estimator is None:
            self._estimator = self.model(self.lambda_s, self.lambda_p)

        fit_args = {**self.kwargs}
        if kwargs:
            fit_args.update(kwargs)

        self._estimator.fit(X, y, **fit_args)
        return self

    def predict(self, X, **kwargs):
        if self._estimator is None:
            raise ValueError("The model has not been trained yet")
        return self._estimator.predict(X, **kwargs)

    def get_params(self, deep=True):
        params = {
            'model': self.model,
            'lambda_s': self.lambda_s,
            'lambda_p': self.lambda_p,
        }
        if deep:
            params.update(self.kwargs)
        return params

    def set_params(self, **params):
        for key, value in params.items():
            if key in self.get_params(deep=False):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        self._estimator = None
        return self

# Set matplotlib backend
matplotlib.use('TkAgg')

# Load data
file_path = r'ps_usable_hydrogen_storage_capacity_gcmcv2.xlsx'
data = pd.read_excel(file_path)

# Data preprocessing (12:ug 13:uv)
XI = data.iloc[:, 5:12]
X = data.iloc[:, 5:12].values
y = data.iloc[:, 13].values.astype(np.float32)

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
X_all = np.concatenate([X_train, X_val, X_test])
y_all = np.concatenate([y_train, y_val, y_test])

# Save scaler
scaler_path = r'scaler_pinn_ps_uv.pkl'
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to {scaler_path}")

# Monotonicity and sample generation
monotonicity_factors = np.array([-1, 1, -1, 1, 1, 0, 0])
monotonic_features = [i for i, factor in enumerate(monotonicity_factors) if factor != 0]

def generate_synthetic_samples(X, num_samples_per_feature=100):
    num_samples = num_samples_per_feature * len(monotonic_features)
    synthetic_X = np.zeros((num_samples, X.shape[1]))

    for i, feature_idx in enumerate(monotonic_features):
        kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(X[:, feature_idx].reshape(-1, 1))
        samples = kde.sample(num_samples_per_feature).flatten()
        if monotonicity_factors[feature_idx] == 1:
            synthetic_X[i * num_samples_per_feature:(i + 1) * num_samples_per_feature, feature_idx] = np.sort(samples)
        elif monotonicity_factors[feature_idx] == -1:
            synthetic_X[i * num_samples_per_feature:(i + 1) * num_samples_per_feature, feature_idx] = np.sort(samples)[::-1]

    non_monotonic_features = [i for i in range(X.shape[1]) if i not in monotonic_features]
    for feature_idx in non_monotonic_features:
        synthetic_X[:, feature_idx] = np.random.choice(X[:, feature_idx], num_samples)

    synthetic_monotonicity = np.tile(monotonicity_factors[monotonic_features], (num_samples, 1))
    return synthetic_X, synthetic_monotonicity

synthetic_X_train, synthetic_monotonicity_train = generate_synthetic_samples(X_train)
synthetic_X_val, synthetic_monotonicity_val = generate_synthetic_samples(X_val)

# Model definition
def build_model(lambda_s=0.001, lambda_p=0.1):
    model = Sequential([
        InputLayer(input_shape=(7,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    def physical_inconsistency_loss(model, synthetic_inputs, monotonicity_factors):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(synthetic_inputs)
            synthetic_outputs = model(synthetic_inputs, training=True)
        grad_synthetic_outputs = tape.gradient(synthetic_outputs, synthetic_inputs)
        physical_loss = 0
        for j in range(monotonicity_factors.shape[1]):
            m_j = monotonicity_factors[:, j]
            gradient_j = grad_synthetic_outputs[:, j]
            physical_loss += tf.reduce_sum((1 - m_j * tf.tanh(gradient_j)) / 2)
        return physical_loss

    def custom_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        regression_loss = tf.reduce_sum(tf.square(y_true - y_pred))
        physical_loss = physical_inconsistency_loss(model,
                                                    tf.convert_to_tensor(synthetic_X_train, dtype=tf.float32),
                                                    tf.convert_to_tensor(synthetic_monotonicity_train,
                                                                         dtype=tf.float32))
        structural_loss = sum(tf.nn.l2_loss(w) for w in model.trainable_variables)
        return regression_loss + lambda_p * physical_loss + lambda_s * structural_loss

    model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_loss, metrics=['mse'])
    return model

model_save_path = r'pinn_ps_uv_constrain.h5'
hyperparams_path = r'hyperparameter_uv.json'

# Custom logger callback
class CustomLogger(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        train_predict = self.model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_predict))
        train_mae = mean_absolute_error(y_train, train_predict)
        train_r2 = r2_score(y_train, train_predict)

        val_predict = self.model.predict(self.validation_data[0])
        val_rmse = np.sqrt(mean_squared_error(self.validation_data[1], val_predict))
        val_mae = mean_absolute_error(self.validation_data[1], val_predict)
        val_r2 = r2_score(self.validation_data[1], val_predict)

        print(f'Epoch {epoch + 1}: '
              f'Train RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f} - '
              f'Validation RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²={val_r2:.4f}')

# Improved model loading logic
force_retrain = False  # Set a flag to control whether to retrain the model
if os.path.exists(model_save_path) and not force_retrain:
    print("Loading existing model...")
    try:
        best_model = load_model(model_save_path, custom_objects={'custom_loss': build_model().loss})
        print("Model loaded successfully!")

        # Load hyperparameters
        if os.path.exists(hyperparams_path):
            with open(hyperparams_path, 'r') as f:
                hyperparams = json.load(f)
                best_lambda_s = hyperparams.get('lambda_s', None)
                best_lambda_p = hyperparams.get('lambda_p', None)
                print(f'Best lambda_s: {best_lambda_s:.4f}')
                print(f'Best lambda_p: {best_lambda_p:.4f}')
        else:
            print("Hyperparameter file not found.")
            best_lambda_s = 0.2  # Default value
            best_lambda_p = 0.2  # Default value

        # Evaluate loaded model
        y_train_pred = best_model.predict(X_train, verbose=0)
        train_rmse = np.sqrt(mean_squared_error(y_train.astype(np.float32), y_train_pred.astype(np.float32)))
        train_mae = mean_absolute_error(y_train.astype(np.float32), y_train_pred.astype(np.float32))
        train_r2 = r2_score(y_train.astype(np.float32), y_train_pred.astype(np.float32))

        y_val_pred = best_model.predict(X_val, verbose=0)
        val_rmse = np.sqrt(mean_squared_error(y_val.astype(np.float32), y_val_pred.astype(np.float32)))
        val_mae = mean_absolute_error(y_val.astype(np.float32), y_val_pred.astype(np.float32))
        val_r2 = r2_score(y_val.astype(np.float32), y_val_pred.astype(np.float32))

        y_test_pred = best_model.predict(X_test, verbose=0)
        test_rmse = np.sqrt(mean_squared_error(y_test.astype(np.float32), y_test_pred.astype(np.float32)))
        test_mae = mean_absolute_error(y_test.astype(np.float32), y_test_pred.astype(np.float32))
        test_r2 = r2_score(y_test.astype(np.float32), y_test_pred.astype(np.float32))

        print("Training set evaluation results:")
        print(f'RMSE={train_rmse:.4f}, MAE={train_mae:.4f}, R²={train_r2:.4f}')
        print("Validation set evaluation results:")
        print(f'RMSE={val_rmse:.4f}, MAE={val_mae:.4f}, R²={val_r2:.4f}')
        print("Test set evaluation results:")
        print(f'RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, R²={test_r2:.4f}')

    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Deleting existing model file and retraining...")
        os.remove(model_save_path)
        force_retrain = True  # Force retraining if loading fails

# If model file does not exist or loading fails, train a new model
if not os.path.exists(model_save_path) or force_retrain:
    print("Training a new model...")
    # Use custom KerasRegressor wrapper
    keras_model = CustomKerasRegressor(
        model=build_model,
        lambda_s=0.001,
        lambda_p=0.1,
        verbose=0
    )

    # Define hyperparameter search space
    param_space = {
        'lambda_s': Real(0.001, 0.1, prior='log-uniform'),
        'lambda_p': Real(0.265, 0.35, prior='log-uniform')
    }

    # Use Bayesian optimization for hyperparameter search
    opt = BayesSearchCV(
        estimator=keras_model,
        search_spaces=param_space,
        n_iter=25,
        scoring='neg_mean_squared_error',
        cv=4,
        n_jobs=1,
        return_train_score=True
    )

    opt.fit(X_train, y_train, callback=VerboseCallback(n_total=100))
    print("Best parameters found:", opt.best_params_)
    print("Best validation RMSE: ", np.sqrt(-opt.best_score_))

    # Rebuild and train model with best hyperparameters
    best_lambda_s = opt.best_params_['lambda_s']
    best_lambda_p = opt.best_params_['lambda_p']

    best_model = build_model(best_lambda_s, best_lambda_p)

    # Save best hyperparameters
    with open(hyperparams_path, 'w') as f:
        json.dump({'lambda_s': best_lambda_s, 'lambda_p': best_lambda_p}, f)

    # Define early stopping and logger callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=150, restore_best_weights=True)
    custom_logger = CustomLogger(validation_data=(X_val, y_val))

    # Train model
    history = best_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=500,
        batch_size=64,
        callbacks=[early_stopping, custom_logger],
        verbose=1
    )

    # Save trained model
    best_model.save(model_save_path)

    # Evaluate model performance
    y_train_pred = best_model.predict(X_train, verbose=0)
    train_rmse = np.sqrt(mean_squared_error(y_train.astype(np.float32), y_train_pred.astype(np.float32)))
    train_mae = mean_absolute_error(y_train.astype(np.float32), y_train_pred.astype(np.float32))
    train_r2 = r2_score(y_train.astype(np.float32), y_train_pred.astype(np.float32))

    y_val_pred = best_model.predict(X_val, verbose=0)
    val_rmse = np.sqrt(mean_squared_error(y_val.astype(np.float32), y_val_pred.astype(np.float32)))
    val_mae = mean_absolute_error(y_val.astype(np.float32), y_val_pred.astype(np.float32))
    val_r2 = r2_score(y_val.astype(np.float32), y_val_pred.astype(np.float32))

    y_test_pred = best_model.predict(X_test, verbose=0)
    test_rmse = np.sqrt(mean_squared_error(y_test.astype(np.float32), y_test_pred.astype(np.float32)))
    test_mae = mean_absolute_error(y_test.astype(np.float32), y_test_pred.astype(np.float32))
    test_r2 = r2_score(y_test.astype(np.float32), y_test_pred.astype(np.float32))

    print("Training set evaluation results:")
    print(f'RMSE={train_rmse:.4f}, MAE={train_mae:.4f}, R²={train_r2:.4f}')
    print("Validation set evaluation results:")
    print(f'RMSE={val_rmse:.4f}, MAE={val_mae:.4f}, R²={val_r2:.4f}')
    print("Test set evaluation results:")
    print(f'RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, R²={test_r2:.4f}')

    print(f'Best lambda_s: {best_lambda_s:.4f}')
    print(f'Best lambda_p: {best_lambda_p:.4f}')

# Plot predictions
def plot_predictions(y_test, y_test_pred, test_r2, test_mae, test_rmse):
    y_test = y_test.astype(np.float32)
    y_test_pred = y_test_pred.flatten().astype(np.float32)

    test_r2_truncated = math.floor(test_r2 * 100) / 100
    test_r2_str = f'{test_r2_truncated:.2f}'
    test_mae_str = f'{test_mae:.2f}'
    test_rmse_str = f'{test_rmse:.2f}'

    plt.figure(figsize=(12, 8))
    plt.scatter(y_test, y_test_pred, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('GCMC calculated gravimetric capacity(g-H$_{2}$ L$^{-1}$)', fontsize=10)
    plt.ylabel('ML predicted gravimetric capacity (g-H$_{2}$ L$^{-1}$)', fontsize=10)
    plt.text(min(y_test), max(y_test_pred),
             f'R$^{{2}}$: {test_r2_str}\nMAE: {test_mae_str}\nRMSE: {test_rmse_str}',
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.savefig('pinn_uv_test.png', dpi=300)
    plt.show()

# Call plot function
plot_predictions(y_test, y_test_pred, test_r2, test_mae, test_rmse)