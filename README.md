#Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions
using SMOTE oversampling, Logistic Regression, and Random Forest.

#Dataset
- **Source:** [Kaggle — ULB Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions | 492 fraud cases (0.17%)
- **Features:** 28 PCA components (V1–V28), Time, Amount

#Tech Stack
- Python, pandas, scikit-learn, imbalanced-learn, matplotlib, seaborn

#What This Project Does
- Handles severe class imbalance (0.17% fraud) using SMOTE oversampling
- Trains Logistic Regression and Random Forest classifiers
- Evaluates using ROC-AUC, Precision, Recall, Confusion Matrix
- Visualizes ROC curves and feature importance

#Results
| Model | ROC-AUC | Recall (Fraud) |
|---|---|---|
| Logistic Regression | ~0.97 | ~91% |
| Random Forest | ~0.99 | ~85% |

#How to Run
1. Clone the repo
   git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection
2. Install dependencies
   pip install -r requirements.txt
3. Add creditcard.csv to the project folder
4. Run the notebook
   jupyter notebook fraud_detection.ipynb
