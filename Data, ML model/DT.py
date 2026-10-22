from sklearn.tree import DecisionTreeRegressor
best_score =0
list_1=np.arange(0,10,1)
list_2=np.arange(1,25,1)
for a in list_1:
    for b in list_2:
            X_train,X_test= train_test_split(X, test_size=0.2,random_state = 6)
            y_train,y_test = train_test_split(y, test_size=0.2,random_state = 6)
            from sklearn.preprocessing  import StandardScaler
            sc = StandardScaler()
            X_train_std = sc.fit_transform(X_train)
            X_test_std = sc.transform(X_test)
        tree = DecisionTreeRegressor(random_state=a,max_depth=b)
        tree.fit(X_train_std,y_train)
        y_train_pred_tree = tree.predict(X_train_std)
        y_test_pred_tree = tree.predict(X_test_std)
        score=r2_score(y_test,y_test_pred_tree)
        if score > best_score:
            best_score = score
            best_parameters = {'random_state': a, 'max_depth': b}
print("Best score: {:.5f}".format(best_score))
print("Best parameters: {}".format(best_parameters))   