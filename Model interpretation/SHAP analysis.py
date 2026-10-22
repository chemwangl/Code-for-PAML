sc = StandardScaler()
X= sc.fit_transform(X)
X1=df.iloc[:,:]
X=pd.DataFrame(X,columns=X1.columns)
import shap
import json 
shap.initjs()
explainer = shap.Explainer(model,X)
shap_values = explainer(X)
shap.summary_plot(shap_values, X,max_display=10)
shap.summary_plot(shap_values, X,max_display=10, 
                  plot_type="bar")
import matplotlib.pyplot as plt
element_names=X.columns.values[:]
shap_values=shap_values.values
shap_values=pd.DataFrame(shap_values,columns=element_names)
#fig, ax = plt.subplots()
plt.figure(figsize=(3,4)) #
feature_name='Hs_wt'
shap_value=shap_values[feature_name]
colors = ['red' if val > 0 else 'blue' for val in shap_value]
a=df[feature_name]

plt.scatter(a,  
            shap_value, 
            c=colors ,s=20)
plt.axvline(x=0.35, color='black', linestyle='--', linewidth=3)
plt.xlabel(feature_name)
plt.ylabel('SHAP value')
plt.title("SHAP value vs. Feature Value")
plt.show()