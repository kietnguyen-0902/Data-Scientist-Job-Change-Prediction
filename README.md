# Data Scientist Job Change Prediction

## 📌 Project Overview
This project builds a complete end-to-end Machine Learning pipeline to predict the job change decisions of Data Scientists. The model's predictions and analytical insights aim to support employee retention strategies and HR analytics[cite: 1].

## 📊 Dataset Overview
- **Total Size:** 19,158 samples.
- **Train Set:** 15,326 samples (80%)[cite: 1].
- **Test Set:** 3,832 samples (20%)[cite: 1].
- **Target Variable:** Binary classification label predicting whether an employee is looking for a new job (`0` = Not looking for a job, `1` = Looking for a job)[cite: 1].
- *Note:* The data is split using a stratified approach to maintain a consistent label ratio across sets, ensuring objective model evaluation[cite: 4].

## ⚙️ Preprocessing Pipeline
The data preprocessing workflow is designed as custom classes integrated directly into a `scikit-learn` `Pipeline` to strictly prevent data leakage:
1. **Missing Value Handling:**
   - Assigns the 'Unknown' or 'No Info' label to categorical missing values (e.g., `company_size`, `company_type`, `major_discipline`)[cite: 2, 3].
   - Imputes the remaining features using the Mode/Median learned exclusively from the Training set[cite: 2, 3].
2. **Feature Encoding:**
   - **Ordinal Encoder:** Applied to hierarchical variables (`education_level`, `experience`, `company_size`, `last_new_job`)[cite: 1, 4].
   - **Frequency Encoder:** Encodes the `city` feature by frequency, converting a high-cardinality categorical variable into a continuous numeric range [0, 1][cite: 1, 4].
   - **One-Hot Encoder:** Applied to nominal categorical variables without an inherent order[cite: 1, 4].
3. **Scaling:** Utilizes `RobustScaler` and a `Log1p` transformation combined with `StandardScaler` to handle skewed variables and extreme outliers, such as training hours (`training_hours`)[cite: 1, 4].

## 🧠 Modeling & Evaluation
The project deployed 4 popular machine learning models. Hyperparameter tuning was performed using `GridSearchCV`[cite: 2]. Additionally, the system automatically identifies the **Optimal Classification Threshold** based on the Precision-Recall curve on the Train set to maximize the **F2-Score** (which is critical for imbalanced datasets prioritizing Recall)[cite: 2, 3].

| Model | Threshold | ROC-AUC (Test) | PR-AUC (Test) | F2-Score (Test) |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | 0.5700[cite: 4] | 0.8203[cite: 4] | 0.5700[cite: 4] | 0.7085[cite: 4] |
| **LightGBM** | 0.3594[cite: 2] | 0.8172[cite: 2] | 0.5587[cite: 2] | 0.7226[cite: 2] |
| **Random Forest** | 0.5551[cite: 3] | 0.8124[cite: 3] | 0.5495[cite: 3] | 0.7151[cite: 3] |
| **Logistic Regression** | Default (0.5000)[cite: 1] | 0.8044[cite: 1] | 0.5481[cite: 1] | 0.7085[cite: 1] |

*The LightGBM model achieved the highest stability and F2-Score on the Test set after threshold tuning.*

## 📈 Feature Importance
Based on feature importance scores (derived from Logistic Regression coefficients and Tree-based models), the key factors influencing the decision to change jobs include[cite: 1]:
* **Retention Factors (Decreased probability of leaving):**
  * Working in regions with a high city development index (`city_development_index`)[cite: 1].
  * Employees working at funded startups (`company_type_Funded Startup`)[cite: 1].
  * Candidates who do not provide clear information about their university major (`major_discipline_No Info`)[cite: 1].
* **Risk Factors (Increased probability of leaving):**
  * Employees from companies with an unknown size (`company_size_Unknown`)[cite: 1].
  * Candidates with degrees related to Business (`major_discipline_Business Degree`)[cite: 1].
  * Working in specific `city` environments that have high turnover frequencies[cite: 1].
