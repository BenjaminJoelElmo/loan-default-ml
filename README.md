# Loan Default Prediction with AutoML and XAI

## Overview
This project compares baseline models (Logistic Regression, Random Forest), tuned Random Forest, and H2O AutoML ensembles for predicting loan defaults.  
Explainable AI (SHAP) is applied to interpret feature importance and interactions.

## Models Evaluated
- Logistic Regression
- Random Forest (untuned)
- Random Forest (tuned with GridSearchCV)
- H2O AutoML Ensemble

## Key Results
- Baselines: High accuracy (~84%) but recall for defaults <5%.
- Tuned RF: Slight ROC-AUC improvement (~0.678) but recall still ~1%.
- AutoML: ROC-AUC ~0.693, recall for defaults ~55% at optimal threshold.
- SHAP: FICO score, interest rate, and credit policy were key drivers of predictions.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Launch Jupyter Notebook: `NextGen_Datascienceproject1.ipynb` and download `loan_data.csv`
3. Run cells sequentially to reproduce results.
