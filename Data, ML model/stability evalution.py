import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



results = []

for random_state in range(100):
    
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=random_state)
    y_train, y_test = train_test_split(y, test_size=0.2, random_state=random_state)

    # 数据标准化
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    # 创建模型并训练
    model =GradientBoostingRegressor(random_state=0,n_estimators=91,max_depth =5)
    model.fit(X_train_std, y_train)

    # 预测
    y_train_pred = model.predict(X_train_std)
    y_test_pred = model.predict(X_test_std)

    # 计算评估指标
    MSE_train = mean_squared_error(y_train, y_train_pred)
    MSE_test = mean_squared_error(y_test, y_test_pred)
    MAE_train = mean_absolute_error(y_train, y_train_pred)
    MAE_test = mean_absolute_error(y_test, y_test_pred)
    R2_train = r2_score(y_train, y_train_pred)
    R2_test = r2_score(y_test, y_test_pred)

    # 保存结果
    results.append({
        'random_state': random_state,
        'MSE_train': MSE_train,
        'MSE_test': MSE_test,
        'MAE_train': MAE_train,
        'MAE_test': MAE_test,
        'R2_train': R2_train,
        'R2_test': R2_test
    })

# 将结果转换为 DataFrame 并打印
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv('100thtsxgb.csv')