"""ML Lab 5 : K-Nearest Neighbor (KNN) for Classification and Regression
Name: Harshit Widhwani | Section: B | Batch: B4 | Roll No: 49
Aim: To implement K-Nearest Neighbor (KNN) algorithm for Classification and Regression.
Same pipeline as b-b4-49_prac5_ml.ipynb, written as a plain Python script.
"""


# ---------------- Part A: KNN Classification (Pima Indians Diabetes dataset) ----------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (train_test_split, GridSearchCV, cross_val_score, StratifiedKFold)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.datasets import load_diabetes
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report,
                             mean_absolute_error, mean_squared_error, r2_score)
import warnings
warnings.filterwarnings("ignore")
print(sns.set_theme(style="whitegrid"))


# ---------------- Load and explore the dataset ----------------
df=pd.read_csv("diabetes.csv")
print(df.head())

print(df.tail())

df.info()

print(df.describe())

print(df.isnull().sum())

plt.figure(figsize=(8,6))
correlation=df.corr()
sns.heatmap(correlation,annot=True,fmt=".2f",cmap="coolwarm",linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()


# ---------------- Data preprocessing ----------------
zero_columns=["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
zero_counts=(df[zero_columns]==0).sum()
print("Zero columns")
print(zero_counts)

df_clean=df.copy()
df_clean[zero_columns]=df_clean[zero_columns].replace(0,np.nan)
print("Missing values after replacing zeros:")
print(df_clean.isnull().sum())

for col in zero_columns:
  df_clean[col]=df_clean[col].fillna(df_clean[col].median())
print("Missing values after median imputation:")
print(df_clean.isnull().sum())

X=df_clean.drop("Outcome",axis=1)
y=df_clean["Outcome"]

print("X shape:",X.shape)
print("Y shape:",y.shape)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("Training samples:",X_train.shape[0])
print("Testing samples:",X_test.shape[0])

print("Training distribution")
print(y_train.value_counts(normalize=True))

print("Testing distribution")
print(y_test.value_counts(normalize=True))

scalar=StandardScaler()
X_train_scaled=scalar.fit_transform(X_train)
X_test_scaled=scalar.transform(X_test)

comparison=pd.DataFrame({
    "Original mean":X_train.mean(),
    "Original std":X_train.std(),
    "Scaled mean":X_train_scaled.mean(axis=0),
    "Scaled std":X_train_scaled.std(axis=0)
})
print(comparison.round(3))


# ---------------- KNN classifier (K = 5) ----------------
K=5
knn=KNeighborsClassifier(n_neighbors=K)
print(knn.fit(X_train_scaled,y_train))

y_pred=knn.predict(X_test_scaled)
print("Predictions:")
print(y_pred[:20])

y_prob=knn.predict_proba(X_test_scaled)[:,1]
print("Predicted probability of Diabetes:")
print(y_prob[:20])

accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.4f}")
precision=precision_score(y_test,y_pred)
print(f"Precision: {precision:.4f}")
recall=recall_score(y_test,y_pred)
print(f"Recall: {recall:.4f}")
f1=f1_score(y_test,y_pred)
print(f"F1 score: {f1:.4f}")

print(classification_report(y_test,y_pred,target_names=["No Diabetes","Diabetes"]))

cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(7,6))

sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=["No diabetes","Diabetes"],yticklabels=["No diabetes","Diabetes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("KNN Confusion matrix")
plt.show()


# ---------------- Choosing the best value of K ----------------
k_values=range(1,31)
train_accuracy=[]
test_accuracy=[]
for k in k_values:
  model=KNeighborsClassifier(n_neighbors=k)
  model.fit(X_train_scaled,y_train)
  train_accuracy.append(model.score(X_train_scaled,y_train))
  test_accuracy.append(model.score(X_test_scaled,y_test))

plt.figure(figsize=(10,6))
plt.plot(k_values,train_accuracy,marker="o",label="Training accuracy")
plt.plot(k_values,test_accuracy,marker="o",label="Testing accuracy")
plt.xlabel("Number of Neighbors(K)")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs K")
plt.xticks(k_values)
plt.legend()
plt.show()

best_k_accuracy=k_values[np.argmax(test_accuracy)]
best_accuracy=max(test_accuracy)
print("Best K:",best_k_accuracy)
print("Best Accuracy:",best_accuracy)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

cv_scores = []

for k in range(1, 31):

    model = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(
        model, X_train_scaled, y_train, cv=cv, scoring="accuracy"
    )

    cv_scores.append(scores.mean())

plt.figure(figsize=(10,6))
plt.plot(range(1, 31), cv_scores, marker="o")
plt.xlabel("Number of Neighbors(K)")
plt.ylabel("Mean CV Accuracy")
plt.title("KNN: K vs 10-fold Cross-validation Accuracy")
plt.xticks(range(1, 31))
plt.show()

best_k_cv=np.argmax(cv_scores)+1
print("Best K based on cross validation:",best_k_cv)
print("Best cv Accuracy:",max(cv_scores))


# ---------------- Hyperparameter tuning using GridSearchCV ----------------
param_grid={
    "n_neighbors":list(range(3,31,2)),"weights":["uniform","distance"],
    "metric":["euclidean","manhattan","minkowski"],
    "p":[1,2]
}

grid_search=GridSearchCV(estimator=KNeighborsClassifier(),param_grid=param_grid,cv=5,scoring="accuracy",n_jobs=-1)
grid_search.fit(X_train_scaled,y_train)
print("Grid Search completed")

print("Best parameters:")
print(grid_search.best_params_)

print("Best Cross-validation Accuracy:",grid_search.best_score_)

best_knn=grid_search.best_estimator_
print("Test Accuracy of tuned KNN:",best_knn.score(X_test_scaled,y_test))


# ---------------- Part B: KNN Regression (Scikit-learn Diabetes dataset) ----------------
diabetes=load_diabetes(as_frame=True)
df_reg=diabetes.frame
print("Shape:",df_reg.shape)
print(df_reg.head())

X_reg=df_reg.drop("target",axis=1)
y_reg=df_reg["target"]

Xr_train,Xr_test,yr_train,yr_test=train_test_split(X_reg,y_reg,test_size=0.2,random_state=42)

reg_scaler=StandardScaler()
Xr_train_scaled=reg_scaler.fit_transform(Xr_train)
Xr_test_scaled=reg_scaler.transform(Xr_test)
print("Training samples:",Xr_train.shape[0])
print("Testing samples:",Xr_test.shape[0])

knn_reg=KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(Xr_train_scaled,yr_train)
yr_pred=knn_reg.predict(Xr_test_scaled)

print("KNN Regression (K=5)")
print(f"MAE      : {mean_absolute_error(yr_test,yr_pred):.4f}")
print(f"MSE      : {mean_squared_error(yr_test,yr_pred):.4f}")
print(f"RMSE     : {np.sqrt(mean_squared_error(yr_test,yr_pred)):.4f}")
print(f"R2 Score : {r2_score(yr_test,yr_pred):.4f}")

cv_r2=[]
for k in range(1,31):
  model=KNeighborsRegressor(n_neighbors=k)
  scores=cross_val_score(model,Xr_train_scaled,yr_train,cv=5,scoring="r2")
  cv_r2.append(scores.mean())

best_k_reg=np.argmax(cv_r2)+1
plt.figure(figsize=(10,6))
plt.plot(range(1,31),cv_r2,marker="o")
plt.xlabel("Number of Neighbors(K)")
plt.ylabel("Mean CV R2 Score")
plt.title("KNN Regression: K vs 5-fold Cross-validation R2")
plt.xticks(range(1,31))
plt.show()
print("Best K for regression:",best_k_reg)
print("Best CV R2 Score:",max(cv_r2))

final_reg=KNeighborsRegressor(n_neighbors=best_k_reg)
final_reg.fit(Xr_train_scaled,yr_train)
yr_best=final_reg.predict(Xr_test_scaled)
print(f"Test MAE  : {mean_absolute_error(yr_test,yr_best):.4f}")
print(f"Test RMSE : {np.sqrt(mean_squared_error(yr_test,yr_best)):.4f}")
print(f"Test R2   : {r2_score(yr_test,yr_best):.4f}")

plt.figure(figsize=(7,6))
plt.scatter(yr_test,yr_best,alpha=0.7)
plt.plot([yr_test.min(),yr_test.max()],[yr_test.min(),yr_test.max()],"r--",label="Perfect prediction")
plt.xlabel("Actual disease progression")
plt.ylabel("Predicted disease progression")
plt.title(f"KNN Regression: Actual vs Predicted (K={best_k_reg})")
plt.legend()
plt.show()
