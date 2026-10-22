import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Set save directory
save_dir = "./results"
os.makedirs(save_dir, exist_ok=True)

# 1. Data splitting and standardization (assuming X and y are already defined)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=6)
y_train, y_test = train_test_split(y, test_size=0.2, random_state=6)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# 2. Define GPR model and grid parameter space
param_grid = {
    'kernel': [1.0 * RBF(1.0), C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))],
    'alpha': [1e-10, 1e-3, 1e-2, 0.1, 1.0], 
    'normalize_y': [True, False]
}

# 3. Manual grid search (without cross-validation)
print(f"\n{'='*40}\nTraining and searching for best GPR parameters...\n{'='*40}")
base_model = GaussianProcessRegressor(random_state=0)
best_score = -float('inf')
best_params = None

for params in ParameterGrid(param_grid):
    model = base_model.set_params(**params)
    model.fit(X_train_std, y_train)
    score = r2_score(y_test, model.predict(X_test_std))
    
    if score > best_score:
        best_score = score
        best_params = params
        
print(f"[Best GPR Parameters]: {best_params}")

# 4. Retrain final model with best parameters and predict
best_model = base_model.set_params(**best_params)
best_model.fit(X_train_std, y_train)

y_train_pred = best_model.predict(X_train_std)
y_test_pred = best_model.predict(X_test_std)

# 5. Calculate evaluation metrics and save to TXT
MSE_train = mean_squared_error(y_train, y_train_pred)
MSE_test = mean_squared_error(y_test, y_test_pred)
MAE_train = mean_absolute_error(y_train, y_train_pred)
MAE_test = mean_absolute_error(y_test, y_test_pred)
R2_train = r2_score(y_train, y_train_pred)
R2_test = r2_score(y_test, y_test_pred)

print(f'MSE_train: {MSE_train:.4f} | MSE_test: {MSE_test:.4f}')
print(f'MAE_train: {MAE_train:.4f} | MAE_test: {MAE_test:.4f}')
print(f'R^2_train: {R2_train:.4f} | R^2_test:  {R2_test:.4f}')

report_path = os.path.join(save_dir, "GPR_Metrics_and_Params.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("========== GPR Model Results ==========\n\n[Best Parameters]:\n")
    for k, v in best_params.items():
        f.write(f"  {k}: {v}\n")
    f.write(f"\n[Performance Metrics]:\n  MSE_train: {MSE_train:.6f}\n  MSE_test : {MSE_test:.6f}\n")
    f.write(f"  MAE_train: {MAE_train:.6f}\n  MAE_test : {MAE_test:.6f}\n")
    f.write(f"  R^2_train: {R2_train:.6f}\n  R^2_test : {R2_test:.6f}\n")

# 6. Save experimental vs predicted comparison values to CSV
df_train = pd.DataFrame({'Dataset': 'Train', 'Experimental': np.ravel(y_train), 'Predicted': np.ravel(y_train_pred)})
df_test = pd.DataFrame({'Dataset': 'Test', 'Experimental': np.ravel(y_test), 'Predicted': np.ravel(y_test_pred)})
pd.concat([df_train, df_test], ignore_index=True).to_csv(os.path.join(save_dir, "GPR_Predictions.csv"), index=False, encoding="utf-8-sig")

# 7. Plotting
plt.figure(figsize=(5,3)) 
plt.scatter(y_train, y_train_pred, color='red', marker='d', edgecolor='white', label='Training data') 
plt.scatter(y_test, y_test_pred, color='blue', marker='s', edgecolor='white', label='Test data')
plt.ylabel('Predicted')
plt.xlabel('Experimental')
plt.title('GPR: Comparison between experimental and predicted')
plt.legend(loc='upper left') 
plt.plot([0.3, 1.8], [0.3, 1.8], c='black', linestyle='-')
plt.xlim([0.3, 1.8]) 
plt.ylim([0.3, 1.8])
plt.tight_layout() 
# plt.savefig(os.path.join(save_dir, "ILCO2_GPR_pred.png"), dpi=300)
plt.show()

print(f"-> GPR results saved to: {os.path.abspath(save_dir)}\n")