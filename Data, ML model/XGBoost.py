import xgboost as xgb
best_score =0
list_1=np.arange(1,21,2)#  
list_2=np.arange(20,40,1)# 
list_3=np.arange(10,150,10)  
for a in list_1:
    for b in list_2:
        for c in list_3:
            X_train,X_test= train_test_split(X, test_size=0.2,random_state = 6)
            y_train,y_test = train_test_split(y, test_size=0.2,random_state = 6)

            #print('X_train shape',X_train.shape,'X_test. shape',X_test.shape)
            #print('y_train shape',y_train.shape,'y_test shape',y_test.shape)

            from sklearn.preprocessing  import StandardScaler
            sc = StandardScaler()
            X_train_std = sc.fit_transform(X_train)
            X_test_std = sc.transform(X_test)
            model=xgb.XGBRegressor(random_state=0,max_features=b,n_estimators=c,max_depth=a)
            model.fit(X_train_std,y_train)
            y_train_pred = model.predict(X_train_std)
            y_test_pred = model.predict(X_test_std)
            score=r2_score(y_test,y_test_pred)
            if score > best_score:
                best_score = score
                best_parameters = {'n_estimators': a, 'max_depth': b,'c':c}
print("Best score: {:.5f}".format(best_score))
print("Best parameters: {}".format(best_parameters)) 