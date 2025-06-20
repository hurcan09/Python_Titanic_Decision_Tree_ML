# 🛳️ Titanic Survival Prediction Project
![titanic](https://github.com/hurcan09/Python_Titanic_Decision_Tree_ML/raw/master/titanic_image.jpeg)

This repository contains two separate Jupyter Notebooks for the Titanic dataset analysis, focusing on data cleaning, feature engineering, and machine learning model evaluation. The goal is to predict passenger survival with improved preprocessing and modeling strategies.

---

## 📦 Libraries Used

```python
pandas, numpy, matplotlib, seaborn, sklearn
```

---

## 🧹 Data Cleaning & Preprocessing

Each notebook begins with cleaning the Titanic dataset, specifically addressing missing data and irrelevant features:

### Notebook 1: `titanic.ipynb`

- **Missing Values**:
  - `age`: Filled using the **median**.
  - `survived`: Filled using the **median** (possibly for exploratory consistency).
  - `embarked`: Filled using the **mode**.
  - `cabin`: Column removed due to excessive missing data.

- **Dropped Columns**: 
  - `cabin`

### Notebook 2: `titanic_v2.ipynb`

- **Missing Values**:
  - `age`: Filled using **median**.
  - `fare`: Filled with median or average fare of similar `Cabin_Code` passengers.
  - `embarked`: Filled using **mode**.
  - `cabin`: Used for extracting `Cabin_Code`, then dropped.
  - Custom logic applied for some categorical transformations (e.g. `boat`, `Cabin_Category`).

- **Dropped Columns**:
  - `body`, `home.dest`, `Cabin_Code`

---

## ⚙️ Feature Engineering

### Notebook 1 Highlights:

- Mapped `sex` to numeric: `{male: 0, female: 1}`
- Created categorical feature from `age`
- Extracted and mapped titles from passenger names

### Notebook 2 Highlights:

- Extracted `Cabin_Code` (first letter of cabin)
- Used `apply()` functions to:
  - Impute missing fare by cabin group
  - Create `Title`, `FareCategory`, `AgeCategory`, `FamilySizeGroup`, `Cabin_Category`
  - Categorize `boat` and map frequent values
- Mapped multiple categorical values to numeric (e.g., `embarked`, `sex`, `title`)
- Removed rare categories and grouped others into `'Other'`

---

## 🤖 Machine Learning Comparison

Two different machine learning models were evaluated to predict survival on the Titanic dataset. Below is a comparison based on their classification reports and confusion matrices:

| Metric       | **RandomForestClassifier** | **DecisionTreeClassifier** |
|--------------|----------------------------|-----------------------------|
| Accuracy     | **0.85**                   | 0.78                        |
| Precision    | 0.81 (class 0) / 0.93 (class 1) | 0.77 (class 0) / 0.82 (class 1) |
| Recall       | 0.96 (class 0) / 0.71 (class 1) | 0.89 (class 0) / 0.63 (class 1) |
| F1-score     | 0.88 (class 0) / 0.80 (class 1) | 0.82 (class 0) / 0.71 (class 1) |
| Confusion Matrix | `[[144  6], [33 79]]`     | `[[134 16], [41 71]]`         |

### ✅ Key Insights:

- The **RandomForestClassifier** outperforms the **DecisionTreeClassifier** in **accuracy** and **f1-score**, especially for detecting class `1` (survivors).
- **RandomForest** shows better balance in precision and recall, making it a more robust model for this classification task.
- **DecisionTree** is simpler and easier to interpret but may overfit or underperform with limited feature engineering.

---

## 🧾 Survival Predictions (Sample)

| Passenger ID | Actual Survival | Predicted |
|--------------|------------------|-----------|
| 100          | 1                | 1         |
| 101          | 0                | 0         |
| 102          | 1                | 0         |
| ...          | ...              | ...       |

---

## 📈 Conclusion

- The second notebook provides a more thorough pipeline with domain-specific feature engineering.
- Custom categorical transformations and conditional imputation improved interpretability.
- The comparison suggests that thoughtful preprocessing impacts model performance as much as the choice of algorithm.

