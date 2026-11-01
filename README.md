# 📊 Salary Prediction Machine Learning Model

A complete end-to-end Machine Learning project that predicts professional salaries using regression techniques, built with Python, Pandas, and Scikit-Learn.

---

## 🚀 Project Overview
This project explores **Supervised Machine Learning** (specifically Linear Regression) using a real-world professional dataset. It covers the complete data science pipeline—from raw data cleaning and exploratory data analysis to feature engineering, rare category grouping, and model evaluation.

---

## 📂 Dataset Features
The model takes various professional attributes into account to predict compensation:
* **`Age`**: Age of the professional.
* **`Gender`**: Gender identity.
* **`Education Level`**: Highest completed education degree (Ordinal encoded).
* **`Job Title`**: Professional role (Optimized via Frequency Thresholding and One-Hot Encoding).
* **`Years of Experience`**: Total professional experience in years (Primary numerical driver).
* **`Salary`**: Target variable (Continuous numerical output).

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.13+
* **Data Manipulation & Cleaning:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (`LinearRegression`, `StandardScaler`, `LabelEncoder`)
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Jupyter Notebook / VS Code

---

## 📈 Workflow Pipeline
1. **Data Cleaning:** Handled missing values (imputing numerical columns with mean and dropping empty target rows).
2. **Frequency Thresholding:** Grouped infrequent job titles into an `"Other"` category to prevent high-cardinality dimensionality explosion.
3. **Feature Encoding:** 
   * Applied **Label Encoding** to ordinal features (`Education Level`).
   * Applied **One-Hot Encoding** (`pd.get_dummies`) with `drop_first=True` to nominal features (`Gender`, `Job Title`).
4. **Model Training:** Fitted a Multiple Linear Regression model using Scikit-Learn.
5. **Evaluation:** Assessed model accuracy using $R^2$ Score, Root Mean Squared Error (RMSE), and Actual vs. Predicted plots.
