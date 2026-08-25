Loan Default Risk Analysis

Final project for a Business Analytics course, McIntire School of Commerce (Fall 2025).

Problem
Given a bank's historical loan dataset, the goal was to predict whether a loan is likely to default and use that prediction to inform a recommendation on whether to sell a given loan to a third party.

Approach
Data cleaning & preparation

Loaded and inspected the raw loan dataset (~870K rows)
Dropped columns with more than 50% missing values and rows with too few non-null fields
Imputed remaining missing values (median for numeric columns, mode for categorical columns)
Removed leakage-prone and irrelevant columns (e.g., post-outcome payment fields, IDs, free-text fields)
Converted the categorical loan_status field into a binary target (default_risk), based on which statuses count as default (Charged Off, Late, etc.)
Balanced the dataset by sampling an equal number of default vs. non-default cases, since defaults were a small minority of the data
Encoded categorical features (grade, term, home ownership, verification status, purpose, etc.) into numeric form using mapping dictionaries and ordinal encoding

Modeling Trained and evaluated several classifiers on the cleaned, balanced dataset:
Decision Tree
K-Nearest Neighbors
Neural Network (MLPClassifier)
Logistic Regression
Support Vector Machine

Then compared ensemble methods to see if they improved on the individual models:
Voting Classifier (hard voting across all five base models)
Bagging
Random Forest
AdaBoost
XGBoost

Each model was evaluated on accuracy, precision, recall, and F1-score, and feature importances were extracted from the tree-based models to identify which loan attributes were most predictive of default (outstanding principal and last payment amount were consistently the strongest predictors).

Segmentation Applied K-Means clustering (selecting k via the elbow method) to group loans into borrower personas based on shared characteristics, to make the recommendations more interpretable for a non-technical audience.

Libraries used
pandas, numpy: data manipulation
seaborn, matplotlib: visualization
scikit-learn: preprocessing, all classification models, ensemble methods, clustering, and evaluation metrics
xgboost: gradient boosting model
statsmodels: used for logistic regression coefficient statistics
