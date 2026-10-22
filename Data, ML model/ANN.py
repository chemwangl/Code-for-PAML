import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split  # Split dataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # Model evaluation
from sklearn.preprocessing import MinMaxScaler
import os
import random

import tensorflow as tf
from tensorflow.keras import optimizers, backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import ParameterGrid


# Split dataset
X_train, X_test = train_test_split(X, test_size=0.2, random_state=6)
y_train, y_test = train_test_split(y, test_size=0.2, random_state=6)

print('X_train shape:', X_train.shape, '| X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape, '| y_test shape:', y_test.shape)

# Standardization
mm1 = MinMaxScaler()
# Use fit_transform to train the scaler and transform simultaneously, then transform the test set
X_train_std = mm1.fit_transform(X_train) 
X_test_std = mm1.transform(X_test)

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # Allocate memory as needed
        print(f"✅ Using GPU: {[g.name for g in gpus]}")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ No GPU detected, using CPU instead")

# Enable deterministic operations (for reproducibility)
try:
    tf.config.experimental.enable_op_determinism()
    print("✅ Deterministic operations enabled")
except AttributeError:
    print("⚠️ Current TF version does not support enable_op_determinism")

# ===================================================

# ================= Save Directory =================
save_dir = "./results1"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# ============================================

input_dim = X_train_std.shape[1]

# 3. Model Building Function
def build_model(layer_sizes, learning_rate, dropout_rate):
    model = Sequential()
    model.add(InputLayer(input_shape=(input_dim,), name="input"))
    for units in layer_sizes:
        model.add(Dense(units, activation='relu'))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='relu'))
    model.compile(loss="mse", optimizer=optimizers.Adam(learning_rate=learning_rate))
    return model

# 4. Callback to keep the best epoch
class KeepBestEpoch(Callback):
    def __init__(self):
        super().__init__()
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.best_epoch = epoch + 1
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)

# 5. Parameter Grid
param_grid = {
   'layer_sizes': [[24, 16], [32, 16], [24, 8], [32, 8], [16, 8], [32, 24]],
   'learning_rate': [0.005, 0.007, 0.009, 0.001, 0.003],
   'dropout_rate': [0.0, 0.1],
   'batch_size': [4, 8, 16, 32, 64]
}

grid = list(ParameterGrid(param_grid))
print(f"\nTotal of {len(grid)} parameter combinations to traverse.")

# 6. Grid Search
best_score = -float('inf')
best_params = None
best_model = None
best_epoch_global = 0
search_history = []

for i, params in enumerate(grid):
    print(f"\nTraining combination {i+1}/{len(grid)}: {params}")

    # ==================== Independent key steps for each search ====================
    # (1) Reset random seed (ensure the same starting point for each training)
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # (2) Clear Keras session, release previous model and GPU memory
    K.clear_session()
    # ============================================================

    model = build_model(
        layer_sizes=params['layer_sizes'],
        learning_rate=params['learning_rate'],
        dropout_rate=params['dropout_rate']
    )

    tracker = KeepBestEpoch()

    model.fit(
        X_train_std, y_train,
        epochs=1000,
        batch_size=params['batch_size'],
        validation_data=(X_test_std, y_test),
        callbacks=[tracker],
        verbose=0
    )

    y_train_pred_current = model.predict(X_train_std, verbose=0)
    y_test_pred_current = model.predict(X_test_std, verbose=0)

    current_train_r2 = r2_score(y_train, y_train_pred_current)
    current_test_r2 = r2_score(y_test, y_test_pred_current)
    current_train_mae = mean_absolute_error(y_train, y_train_pred_current)
    current_test_mae = mean_absolute_error(y_test, y_test_pred_current)

    print(f"  -> Best Epoch: {tracker.best_epoch} | Test R²: {current_test_r2:.4f}")

    search_history.append({
        'Iteration': i + 1,
        'layer_sizes': str(params['layer_sizes']),
        'learning_rate': params['learning_rate'],
        'dropout_rate': params['dropout_rate'],
        'batch_size': params['batch_size'],
        'Best_Epoch': tracker.best_epoch,
        'Train_R2': current_train_r2,
        'Test_R2': current_test_r2,
        'Train_MAE': current_train_mae,
        'Test_MAE': current_test_mae
    })

    if current_test_r2 > best_score:
        best_score = current_test_r2
        best_params = params
        best_model = model
        best_epoch_global = tracker.best_epoch

# 7. Save search results
df_history = pd.DataFrame(search_history)
history_csv_path = os.path.join(save_dir, "ANN_GridSearch_All_Trials.csv")
df_history.to_csv(history_csv_path, index=False, encoding="utf-8-sig")
print(f"\n-> Detailed records of all combinations saved to: {history_csv_path}")
print(f"[ANN Global Best Hyperparameters]: {best_params}")

# 8. Final Evaluation
y_train_pred = best_model.predict(X_train_std, verbose=0)
y_test_pred = best_model.predict(X_test_std, verbose=0)

score_train = r2_score(y_train, y_train_pred)
score_test = r2_score(y_test, y_test_pred)
MSE_train = mean_squared_error(y_train, y_train_pred)
MSE_test = mean_squared_error(y_test, y_test_pred)
MAE_train = mean_absolute_error(y_train, y_train_pred)
MAE_test = mean_absolute_error(y_test, y_test_pred)

print(f'MSE_train: {MSE_train:.4f} | MSE_test: {MSE_test:.4f}')
print(f'MAE_train: {MAE_train:.4f} | MAE_test: {MAE_test:.4f}')
print(f'R^2_train: {score_train:.4f} | R^2_test: {score_test:.4f}')

# 9. Save result files
name = "ANN_GridSearch"
report_path = os.path.join(save_dir, f"{name}_Metrics_and_Params.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"========== {name} Model Results ==========\n\n")
    f.write("[Best Parameters]:\n")
    for k, v in best_params.items():
        f.write(f"  {k}: {v}\n")
    f.write(f"  best_epoch: {best_epoch_global}\n")
    f.write("\n[Performance Metrics]:\n")
    f.write(f"  MSE_train: {MSE_train:.6f}\n")
    f.write(f"  MSE_test : {MSE_test:.6f}\n")
    f.write(f"  MAE_train: {MAE_train:.6f}\n")
    f.write(f"  MAE_test : {MAE_test:.6f}\n")
    f.write(f"  R^2_train: {score_train:.6f}\n")
    f.write(f"  R^2_test : {score_test:.6f}\n")

# 10. Save prediction results
df_train = pd.DataFrame({'Dataset': 'Train', 'Experimental': np.ravel(y_train), 'Predicted': np.ravel(y_train_pred)})
df_test = pd.DataFrame({'Dataset': 'Test', 'Experimental': np.ravel(y_test), 'Predicted': np.ravel(y_test_pred)})
df_all = pd.concat([df_train, df_test], ignore_index=True)
df_all.to_csv(os.path.join(save_dir, f"{name}_Predictions.csv"), index=False, encoding="utf-8-sig")

print(f"\n-> All results saved to: {os.path.abspath(save_dir)}\n")