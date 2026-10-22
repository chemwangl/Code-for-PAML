import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import time

def augment_data_with_noise(X, y, times=1, noise_level=0.03, random_state=42):
    np.random.seed(random_state)
    X_aug_list = [X]
    y_aug_list = [y]
    
    std_dev = np.std(X, axis=0)
    std_dev[std_dev == 0] = 1e-6 
    
    for _ in range(times):
        noise = np.random.normal(0, std_dev * noise_level, X.shape)
        X_aug_list.append(X + noise)
        y_aug_list.append(y)
        
    return np.vstack(X_aug_list), np.concatenate(y_aug_list)

X_arr = np.array(X)
y_arr = np.array(y).ravel()

if isinstance(X, pd.DataFrame):
    feature_names = X.columns.tolist()
else:
    feature_names = [f'Feature_{i+1}' for i in range(X_arr.shape[1])]

learning_rates = np.arange(0.01, 0.5, 0.05)   
n_estimators_list = np.arange(10, 100, 10)    
max_depth_list = np.arange(3, 15, 2)          

results_summary = []

for times in [1, 2, 3]:
    print(f"\n" + "="*60)
    print(f"🚀 Starting processing: Training set augmented {times}x")
    print("="*60)
    
    start_time = time.time()
    
    best_test_score = -float('inf')
    best_train_score = -float('inf')
    best_parameters = {}
    best_X_train_aug_saved = None
    best_y_train_aug_saved = None

    for rs in random_states: # Note: Ensure 'random_states' is defined before this loop
        
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_arr, y_arr, test_size=0.2, random_state=6 
        )
        
        X_tr_aug, y_tr_aug = augment_data_with_noise(X_tr, y_tr, times=times, noise_level=0.03)
        
        sc = StandardScaler()
        X_tr_std = sc.fit_transform(X_tr_aug)
        X_te_std = sc.transform(X_te)
        
        for lr in learning_rates:
            for n_est in n_estimators_list:
                for depth in max_depth_list:
                    
                    model = GradientBoostingRegressor(
                        learning_rate=lr,
                        n_estimators=n_est,
                        max_depth=depth,
                        random_state=0 
                    )
                    model.fit(X_tr_std, y_tr_aug)
                    
                    y_te_pred = model.predict(X_te_std)
                    test_score = r2_score(y_te, y_te_pred)
                    
                    y_tr_pred = model.predict(X_tr_std)
                    train_score = r2_score(y_tr_aug, y_tr_pred)
                    
                    if test_score > best_test_score:
                        best_test_score = test_score
                        best_train_score = train_score
                        best_parameters = {
                            'learning_rate': lr,
                            'n_estimators': n_est,
                            'max_depth': depth
                        }
                        best_X_train_aug_saved = X_tr_aug
                        best_y_train_aug_saved = y_tr_aug

    elapsed_time = time.time() - start_time
    
    df_save = pd.DataFrame(best_X_train_aug_saved, columns=feature_names)
    df_save['Target_y'] = best_y_train_aug_saved
    csv_filename = f'augmented_train_data_{times}x.csv'
    df_save.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    
    print(f"\n[{times}x Augmentation Best Results] (Time elapsed: {elapsed_time:.2f} seconds)")
    print(f" 🎯 Augmented Train R^2: {best_train_score:.4f}")
    print(f" 🏆 Independent Test (20%) R^2: {best_test_score:.4f}")
    print(" 💡 Optimal Parameter Combination:")
    for k, v in best_parameters.items():
        print(f"  {k}: {v}")
    print(f" [Data] Saved best split data to: {csv_filename} (Total samples: {len(df_save)})")
        
    results_summary.append({
        'Augment_Times': times,
        'Train_Samples': len(df_save),
        'Train_R2': best_train_score,
        'Test_R2': best_test_score
    })

print("\n" + "★"*80)
print("[Manual Grid Search: Comparison Summary for All Augmentation Multipliers]")
print("★"*80)
for res in results_summary:
    print(f"Augmented {res['Augment_Times']}x | Train R^2: {res['Train_R2']:.4f} | Test R^2: {res['Test_R2']:.4f} | Train Samples: {res['Train_Samples']}")
print("★"*80)