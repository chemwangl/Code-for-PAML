import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
df=pd.read_csv('data.csv')
X=df.iloc[:,0:49]
y=df.iloc[:,49]

y = df.iloc[:,49].values.ravel()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

# ======================
# 2. 标准化
# ======================
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std  = sc.transform(X_test)

# ======================
# 3. 训练基线模型（不扰动）
# ======================
model =GradientBoostingRegressor(n_estimators=91,max_depth=7,random_state=0,alpha=0.1)
model.fit(X_train_std, y_train)

# ======================
# 4. 基线性能
# ======================
y_test_pred = model.predict(X_test_std)
r2_base = r2_score(y_test, y_test_pred)
mae_base = mean_absolute_error(y_test, y_test_pred)
mse_base = mean_squared_error(y_test, y_test_pred)

print("Baseline performance")
print(f"R² = {r2_base:.4f} | MAE = {mae_base:.4f} | MSE = {mse_base:.4f}")


key_features = ['pieceVSA_EState2','pieceHallKierAlpha','piecePEOE_VSA9','pieceMinEStateIndex'] 
print("\nKey features (manually selected):", key_features)


X_test_perturbed = X_test.copy()
noise_ratio = 0.05  

for col in key_features:
    noise = np.random.normal(
        0,
        noise_ratio * X_test[col].std(),
        size=len(X_test)
    )
    X_test_perturbed[col] += noise


X_test_perturbed_std = sc.transform(X_test_perturbed)


y_test_pred_perturbed = model.predict(X_test_perturbed_std)
r2_perturbed = r2_score(y_test, y_test_pred_perturbed)
mae_perturbed = mean_absolute_error(y_test, y_test_pred_perturbed)
mse_perturbed = mean_squared_error(y_test, y_test_pred_perturbed)

print("\nAfter perturbing key features")
print(f"R² = {r2_perturbed:.4f} | MAE = {mae_perturbed:.4f} | MSE = {mse_perturbed:.4f}")

print("\nPerformance degradation")
print(f"ΔR²  = {r2_base - r2_perturbed:.4f}")
print(f"ΔMAE = {mae_perturbed - mae_base:.4f}")
print(f"ΔMSE = {mse_perturbed - mse_base:.4f}")