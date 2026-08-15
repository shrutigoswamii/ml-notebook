# Telco Customer Churn Prediction

Predicting customer churn for a telecom company using classical ML, with a full
pipeline from raw data to a statistically-validated, tuned final model.

## Problem
Identify customers at risk of churning so the business can prioritize retention
efforts, treating false negatives (missed churners) as more costly than false
positives (unnecessary retention offers) — a recall-prioritized objective.

## Data
Telco Customer Churn dataset (Kaggle) — 7,043 customers, 21 original features
covering demographics, account details, and service usage.

## Approach
1. **EDA** — univariate/bivariate analysis, correlation, class imbalance,
   outlier investigation
2. **Data Cleaning** — resolved a hidden data-quality issue in `TotalCharges`
   (11 rows with blank values, all `tenure=0`; filled with 0 based on business
   logic rather than dropping or mean-imputing)
3. **Feature Engineering** — created `TotalAddons` (count of add-on services),
   validated it shows a strong, clean relationship with churn (46% churn at 1
   addon → 4.7% churn at 6 addons)
4. **Preprocessing** — one-hot/label encoding, StandardScaler (fit on train
   only), stratified 80/20 train/test split, SMOTE/class-weight imbalance handling
5. **Multicollinearity check** — VIF analysis uncovered and fixed 3 hidden
   encoding-artifact redundancies (duplicated "No internet service" categories,
   a double-counted engineered feature, and a redundant phone-service category)
6. **Modeling** — trained and compared 4 models: Logistic Regression, Decision
   Tree, Random Forest, XGBoost
7. **Evaluation** — confusion matrix, precision/recall/F1, ROC-AUC, 5-fold
   stratified cross-validation, hyperparameter tuning (RandomizedSearchCV)

## Key Findings

**Churn risk profile:** customers with month-to-month contracts, high monthly
charges, low tenure, fiber optic internet, and electronic check payment show
substantially elevated churn risk — a consistent "new + expensive + uncommitted"
pattern confirmed across EDA, all 4 models' feature importance, and logistic
regression's odds ratios.

**Model comparison (5-fold cross-validated AUC):**

| Model | Mean AUC | Std Dev |
|---|---|---|
| Decision Tree | 0.824 | 0.009 |
| Random Forest | 0.839 | 0.011 |
| Logistic Regression | 0.843 | 0.011 |
| XGBoost (default) | 0.846 | 0.012 |
| **XGBoost (tuned)** | **0.847** | — |

Decision Tree is reliably the weakest model. Logistic Regression, Random Forest,
and XGBoost are statistically indistinguishable — their cross-validated ranges
overlap substantially. Hyperparameter tuning yielded a negligible improvement
(+0.001), indicating performance is bounded by available feature signal rather
than model choice or tuning.

**Interesting model-architecture disagreement:** Logistic Regression ranks
`MonthlyCharges` as the single strongest churn driver, while tree-based models
(Random Forest, XGBoost) rank `Contract` type far higher and `MonthlyCharges`
far lower. This stems from multicollinearity between the two — trees "use up"
shared information at early, greedy splits, while linear models weight every
feature simultaneously. Both are valid views of correlated features, not a
contradiction.

## Final Model
**XGBoost** (tuned: `max_depth=3, learning_rate=0.05, n_estimators=100,
subsample=0.8, reg_lambda=1, reg_alpha=0.1`) — selected for the best recall
(0.81, prioritized since missing a churner is costlier than an unnecessary
retention offer) and marginally best AUC. Logistic Regression remains a
reasonable, simpler, more interpretable alternative given the small practical
performance gap between the two.

## What I'd Do Next
- Feature engineering: separate "no internet service" from "internet but no
  add-ons chosen" in `TotalAddons` (currently conflates two different populations)
- Try SHAP values for more rigorous, unified feature attribution across models
- Deploy the tuned XGBoost model behind a simple API for real-time scoring

## Tech Stack
Python, pandas, scikit-learn, XGBoost, imbalanced-learn, matplotlib/seaborn