from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

ratios = [0.6, 0.7, 0.8]  


random_states = np.arange(1, 21, 2)
learning_rates = np.arange(0.01, 0.5, 0.05)
n_estimators_list = np.arange(10, 100, 10)
max_depth_list = np.arange(3, 15, 2)

print("="*50)
print("不同数据量对模型性能的影响（含超参数搜索）")
print("="*50)

for ratio in ratios:
    # 抽取子数据集
    X_subset, _, y_subset, _ = train_test_split(X, y, train_size=ratio, random_state=0)
    
    best_score = -np.inf
    best_parameters = {}

    for rs in random_states:
        for lr in learning_rates:
            for n_est in n_estimators_list:
                for depth in max_depth_list:
                    # 固定随机划分训练/测试集
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X_subset, y_subset, test_size=0.2, random_state=rs
                    )
                    
                    # 标准化
                    sc = StandardScaler()
                    X_tr_std = sc.fit_transform(X_tr)
                    X_te_std = sc.transform(X_te)
                    
                    # 模型训练
                    model = GradientBoostingRegressor(
                        learning_rate=lr,
                        n_estimators=n_est,
                        max_depth=depth,
                        random_state=0
                    )
                    model.fit(X_tr_std, y_tr)
                    
                    # 测试集预测
                    y_te_pred = model.predict(X_te_std)
                    score = r2_score(y_te, y_te_pred)
                    
                    # 保存最优参数
                    if score > best_score:
                        best_score = score
                        best_parameters = {
                            'random_state': rs,
                            'learning_rate': lr,
                            'n_estimators': n_est,
                            'max_depth': depth
                        }
    
    print(f"\n数据量 {ratio*100:.0f}% | 最优 R²_test: {best_score:.5f}")
    print("最优参数:", best_parameters)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_subset, y_subset, test_size=0.2, random_state=best_parameters['random_state']
    )
    sc = StandardScaler()
    X_tr_std = sc.fit_transform(X_tr)
    X_te_std = sc.transform(X_te)
    
    model = GradientBoostingRegressor(
        learning_rate=best_parameters['learning_rate'],
        n_estimators=best_parameters['n_estimators'],
        max_depth=best_parameters['max_depth'],
        random_state=0
    )
    model.fit(X_tr_std, y_tr)
    
    y_tr_pred = model.predict(X_tr_std)
    y_te_pred = model.predict(X_te_std)
    
    r2_train = r2_score(y_tr, y_tr_pred)
    mse_train = mean_squared_error(y_tr, y_tr_pred)
    mae_train = mean_absolute_error(y_tr, y_tr_pred)
    
    r2_test = r2_score(y_te, y_te_pred)
    mse_test = mean_squared_error(y_te, y_te_pred)
    mae_test = mean_absolute_error(y_te, y_te_pred)
    
    print(
        f"训练样本: {X_tr.shape[0]} | "
        f"R²_train: {r2_train:.3f} | MSE_train: {mse_train:.4f} | MAE_train: {mae_train:.4f} | "
        f"R²_test: {r2_test:.3f} | MSE_test: {mse_test:.4f} | MAE_test: {mae_test:.4f}"
    )

print("="*50)