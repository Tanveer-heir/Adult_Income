## 📌 Dataset

**Adult Income Dataset** (UCI Machine Learning Repository)  
Also known as the *Census Income Dataset*.

**Task:** Binary classification  
Predict whether an individual earns **>50K** or **≤50K** per year based on demographic and employment features.

- Rows: 32,561  
- Features: 14  
- Target: `income`  
- Class distribution: Imbalanced (~76% ≤50K, ~24% >50K)

---

## 🧠 Project Objectives

- Understand and clean real-world tabular data
- Handle missing values correctly
- Apply appropriate preprocessing to numerical and categorical features
- Prevent data leakage using pipelines
- Build and evaluate baseline classification models
- Go beyond accuracy and use proper evaluation metrics

---

## 🧹 Data Preprocessing

### Missing Values
- Dataset contains missing values encoded as `"?"`
- Converted to `NaN` before preprocessing

### Feature Types
- **Numerical (6):** age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
- **Categorical (8):** workclass, education, marital-status, occupation, relationship, race, sex, native-country

---

## ⚙️ Preprocessing Pipelines

### Numerical Pipeline
- `SimpleImputer(strategy="median")`
- `StandardScaler`

### Categorical Pipeline
- `SimpleImputer(strategy="most_frequent")`
- `OneHotEncoder(handle_unknown="ignore")`

### ColumnTransformer
Numerical and categorical pipelines are combined using `ColumnTransformer` to ensure correct column-wise preprocessing.

---

## 🔗 Pipelines & Data Leakage Prevention

All preprocessing steps are integrated with models using `Pipeline`, ensuring:
- Preprocessing is learned **only from training data**
- No information leaks into the test set
- Clean and reusable ML workflow

---

## 🤖 Models (Baselines)

Three baseline classifiers were trained and evaluated:

1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Naive Bayes (GaussianNB)**  
   - Trained separately due to sparse matrix limitations

---

## 📊 Evaluation Metrics

Because the dataset is **imbalanced**, performance was evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Accuracy alone was not considered sufficient.

---

## 📈 Results Summary

| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression | ~0.86     | ~0.74      | ~0.62   | **~0.67** |
| KNN                | ~0.84     | ~0.68      | ~0.61   | ~0.64     |
| Naive Bayes        | ~0.54     | ~0.34      | ~0.95   | ~0.50     |

### ✅ Best Model
**Logistic Regression** performed best overall, achieving the highest F1-score and the most balanced trade-off between precision and recall.

---

## 🏆 Key Takeaways

- Proper preprocessing is critical for reliable ML models
- Pipelines are essential for preventing data leakage
- Scaling matters for distance-based and linear models
- Accuracy can be misleading on imbalanced datasets
- Logistic Regression is a strong, interpretable baseline for tabular data

---

## 🔜 Next Steps

- Feature engineering
- Hyperparameter tuning
- Tree-based models
- ROC-AUC and threshold tuning
- Model deployment

---

## 📚 Tools & Libraries

- Python
- Pandas, NumPy
- Scikit-learn

---

## 👤 Author

Tanveer Singh  

