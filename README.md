# FinTech Innovations: Machine Learning Model for Loan Approval 🏦📈

**Author:** Abdullahi Yussuf Ibrahim  
**Project Type:** Machine Learning Classification / Risk Analytics  
**Institution:** Moringa School Data Science Program  

## 📝 Project Overview
FinTech Innovations is a growing financial technology company partnering with traditional banks to modernize loan approval processes. Currently, the loan approval process relies heavily on manual review, leading to inconsistent decisions, slower response times, and human bias.

This project develops a robust, data-driven machine learning pipeline to automate and enhance the loan approval decision-making process. By predicting the likelihood of loan default, this model standardizes screening, reduces bias, and identifies creditworthy applicants who traditional criteria might overlook.

## 🎯 Business Problem & Asymmetric Cost Matrix
In this specific business context, model errors have highly unequal financial implications:
* **False Positive (Approving a Bad Loan):** Costs the company **$50,000** in defaults.
* **False Negative (Denying a Good Loan):** Costs the company **$8,000** in lost interest revenue.

**Goal:** Build a classification model that minimizes the *Total Expected Cost* by optimizing for Recall on the default class and ROC-AUC, rather than standard accuracy.

## 🗄️ The Dataset
The model was trained on historical data comprising 20,000 past loan applications. 
* **Target Variable:** `LoanApproved` (Class 0: Denied/Default Risk, Class 1: Approved).
* **Imbalance:** ~76% Denied, ~24% Approved.
* **Features:** A mix of numerical and categorical data, including direct financial indicators (Debt-to-Income Ratio, Credit Score) and behavioral signals.

## 🛠️ Methodology (CRISP-DM Framework)
This project rigorously follows the Cross-Industry Standard Process for Data Mining (CRISP-DM):

1. **Business Understanding:** Defined custom evaluation metrics based on the $50k/$8k asymmetric cost matrix to ensure the model aligns with executive bottom-line goals.
2. **Data Exploration & Analysis (EDA):** Conducted univariate and bivariate analysis, uncovering correlations between income, credit scores, and approval rates while handling the severe class imbalance.
3. **Data Preparation & Pipeline Engineering:** * Developed a **Custom Scikit-Learn Transformer** (`CurrencyCleaner`) to programmatically handle messy string data (e.g., converting `"$39,948.00"` to float).
   * Designed a sophisticated `ColumnTransformer` utilizing parallel pipelines for categorical encoding, numerical scaling, and missing value imputation.
4. **Iterative Modeling:** Trained and evaluated multiple algorithms (Logistic Regression, Random Forest, XGBoost) integrating the preprocessing pipeline directly into the model training loop to prevent data leakage.
5. **Model Evaluation:** Tuned hyperparameters using `GridSearchCV` and translated final model performance into tangible business value (dollar-cost savings compared to a baseline manual approach).

## 💻 Tech Stack
* **Language:** Python 3.8+
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, SciPy
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Jupyter Notebooks

## 🚀 How to Run the Project
1. Clone this repository to your local machine.
2. Ensure you have the required libraries installed: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Open the Jupyter Notebook (`financial_loan_risk.ipynb`).
4. Run the cells sequentially from top to bottom.

## 💡 Future Improvements
* **Feature Engineering:** Explore interaction terms between `DebtToIncomeRatio` and `TotalAssets`.
* **Alternative Thresholding:** Implement dynamic threshold tuning to pinpoint the exact probability cut-off that mathematically minimizes the custom cost function.
