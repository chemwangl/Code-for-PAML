import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor

# 1. Variance Thresholding (Low variance filter)
var_selector = VarianceThreshold(threshold=0.1)
X_var = var_selector.fit_transform(X)

var_cols = X.columns[var_selector.get_support()]
X_var = pd.DataFrame(X_var, columns=var_cols)

# 2. Random Forest Feature Importance Evaluation
rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_var, y)

importance = pd.Series(
    rf.feature_importances_,
    index=X_var.columns
)

# Extract top 100 features based on importance
top100_features = importance.sort_values(ascending=False).head(100).index
X_rf = X_var[top100_features]

# 3. Multicollinearity / Correlation Filtering
sorted_features = importance[top100_features].sort_values(ascending=False).index.tolist()

corr_matrix = X_rf.corr().abs()

selected_features = []

for feature in sorted_features:
    if not selected_features:
        selected_features.append(feature)
    else:
        max_corr = corr_matrix.loc[feature, selected_features].max()
        if max_corr < 0.9:
            selected_features.append(feature)

X_final = X_rf[selected_features]

# ======================
# Final Results
# ======================
print("Final number of features:", X_final.shape[1])
print("Final feature names:")
print(selected_features)