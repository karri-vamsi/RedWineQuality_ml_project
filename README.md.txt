## 🍷 Red Wine Quality Classification (Random Forest & KNN)

This machine learning project classifies red wine quality scores using **Random Forest** and **K-Nearest Neighbors (KNN)** models. It tackles a **multiclass imbalanced classification** problem using chemical property features.

---

### 📌 Project Overview

- **Objective**: Predict red wine quality scores (ranging from 3 to 8) based on physicochemical test results.
- **Approach**: Applied and compared Random Forest and KNN models (with and without hyperparameter tuning).
- **Dataset**: [`winequality-red.csv`](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)  
- **Imbalance**: Class distribution skewed toward quality 5 and 6 wines.

---

### 📁 Repository Contents

| File Name                         | Description                                     |
|-----------------------------------|-------------------------------------------------|
| `RedWineQuality_ML_project.ipynb` | Full notebook with code, EDA, modeling steps    |
| `winequality-red%202.csv`         | Red wine dataset used for training and testing  |
| `Analysis_questions.png`          | Questions provided for report-based evaluation  |


---

### 🔍 Project Workflow

#### 1️⃣ Data Exploration & Preprocessing

- No missing values in the dataset.
- Target variable (`quality`) is a multiclass label from **3 to 8**.
- Used `StratifiedShuffleSplit` to ensure class balance in train/test splits.
- Created:
  - **Raw features** (for Random Forest)
  - **MinMax-scaled features** (for KNN, which is scale-sensitive)

#### 2️⃣ Baseline Models (No Tuning)

| Model           | Accuracy | Weighted F1 Score |
|-----------------|----------|-------------------|
| Random Forest   | 69.0%    | 0.669             |
| KNN             | 61.3%    | 0.594             |

- **Random Forest** performed better in initial evaluation.
- Both models evaluated using `classification_report`, confusion matrices, and cross-validation.

#### 3️⃣ Hyperparameter Tuning (RandomizedSearchCV)

- **Random Forest Parameters Tuned**: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- **KNN Parameters Tuned**: `n_neighbors`, `weights`, `metric`

| Tuned Model      | Accuracy | Weighted F1 Score |
|------------------|----------|-------------------|
| Random Forest    | 66.9%    | 0.644             |
| KNN              | 65.4%    | 0.637             |

- Random Forest still performed slightly better after tuning.
- No drastic gain post tuning — indicates default settings already reasonable.

---

### 📊 Final Model Comparison

| Metric             | Random Forest (Tuned) | KNN (Tuned)    |
|--------------------|------------------------|----------------|
| Accuracy           | 66.9%                 | 65.4%          |
| Weighted F1 Score  | 0.644                 | 0.637          |
| Strengths          | Handles imbalance, feature importance | Simplicity, effective with scaled data |
| Weaknesses         | Slower, complex tuning | Sensitive to noise & scaling |

---

### ✅ Libraries Used

- `pandas`, `matplotlib`
- `sklearn`: preprocessing, model_selection, metrics, ensemble, neighbors

---

### 🧠 Concepts Covered

- Multiclass classification
- Model evaluation with imbalanced data
- Hyperparameter tuning via `RandomizedSearchCV`
- Cross-validation
- Confusion matrix visualization

---

### 👨‍💻 Author

**Karri Vamsi**  
Data Science & Machine Learning Enthusiast  
📅 Project Completed: July 27, 2025
