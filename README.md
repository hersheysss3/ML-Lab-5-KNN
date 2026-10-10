# ML Lab-5 : K-Nearest Neighbor (KNN) for Classification and Regression

**Name:** Harshit Widhwani &nbsp;|&nbsp; **Section:** B &nbsp;|&nbsp; **Roll No:** 49 &nbsp;|&nbsp; **Batch:** B4

Ramdeobaba University, Nagpur | Department of Artificial Intelligence and Machine Learning

---

## Aim

To implement K-Nearest Neighbor (KNN) algorithm for Classification and Regression.

## Dataset

`diabetes.csv`, the Pima Indians Diabetes dataset (768 rows, 9 columns). The regression part uses scikit-learn's built-in `load_diabetes()` dataset.

## Files in this repository

| File | Description |
|---|---|
| [`b-b4-49_prac5_ml.ipynb`](b-b4-49_prac5_ml.ipynb) | **Main notebook with every step and all outputs already executed.** Opens directly in the browser. |
| `knn_classification_regression.py` | The same pipeline as a plain Python script. |
| `diabetes.csv` | The dataset used. |

## Steps applied

1. Loaded `diabetes.csv` and inspected it with `head()`, `info()` and `describe()`
2. Plotted a correlation heat-map of all columns
3. Replaced the invalid zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI with NaN and filled them with the column median
4. Split the data 80 / 20 and standardised the features with `StandardScaler`
5. Trained a **KNeighborsClassifier** with K = 5 and evaluated it with accuracy, precision, recall, F1-score, a classification report and a confusion matrix
6. Plotted training / test accuracy and **10-fold cross-validation** accuracy for K = 1 to 30
7. Tuned `n_neighbors`, `weights`, `metric` and `p` with **GridSearchCV**
8. Built a **KNeighborsRegressor** on `load_diabetes()`, evaluated it with MAE, MSE, RMSE and R², chose K with cross-validated R² and plotted actual vs predicted values

## How to run

```bash
pip install numpy pandas matplotlib seaborn scikit-learn notebook
```

Open the notebook (or run the `.py` script) from inside this folder so the dataset path resolves.
