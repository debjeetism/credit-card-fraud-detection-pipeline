# Credit Card Fraud Detection

Detecting fraudulent transactions in credit card data using machine learning models with advanced techniques to address class imbalance.

---

## 📋 Problem Statement

This project aims to **predict fraudulent credit card transactions** using machine learning and advanced techniques for handling highly imbalanced data. The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and contains 284,807 transactions (only 492 frauds, ~0.17%).

**Business relevance:**  
Credit card fraud costs financial institutions billions annually. The ability to accurately flag fraudulent transactions is critical for banks to protect both their customers and their own bottom line.

---

## 🗂️ Dataset Description

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Rows:** 284,807 transactions over two days (European cardholders)
- **Features:**  
  - `Time`: Seconds from first transaction  
  - `Amount`: Transaction amount  
  - `V1` ... `V28`: PCA-obfuscated features  
  - `Class`: Target (1 = Fraud, 0 = Non-fraud)
- **Class Imbalance:** Only 0.17% transactions are frauds (492 rows).

---

## 🚀 Project Workflow

1. **Exploratory Data Analysis (EDA):**
   - Statistical summaries and visualization of features
   - Temporal patterns and class imbalance exploration
2. **Preprocessing:**
   - Handled skewness, dropped `Time`, feature scaling if needed
   - Split into train/test sets with stratified sampling
3. **Dealing with Class Imbalance:**
   - Random Oversampling
   - SMOTE (Synthetic Minority Oversampling Technique)
   - ADASYN (Adaptive Synthetic)
4. **Model Building & Evaluation:**
   - Trained and tuned: Logistic Regression, k-NN (for small subsets), Decision Tree, SVM, Random Forest, XGBoost, Neural Networks
   - Cross-validated hyperparameters
   - Compared effect of different resampling strategies
5. **Evaluation Metrics:**
   - Main metric: **ROC-AUC**
   - Special attention to minimizing false negatives (catching fraud)
   - Plotted ROC curves, feature importances

---

## 🧑‍💻 Code Structure

All code is written in a single Jupyter notebook for transparency and didactic purposes. Core libraries:
- Python (Pandas, NumPy, Matplotlib, Seaborn)
- scikit-learn
- imbalanced-learn
- XGBoost
- TensorFlow/Keras (for neural nets)

---

## 📈 Key Highlights

- **Full EDA**: Visualized feature distributions and class separations
- **Model comparisons**: Side-by-side AUC metrics across various classifiers
- **Imbalance techniques**: Demonstrated RandomOverSampler, SMOTE, ADASYN with both ensemble and boosting models
- **Best results**:  
    - **Random Forest (SMOTE-oversampled):** ROC-AUC ≈ 0.985  
    - **XGBoost (oversampling):** ROC-AUC ≈ 0.983

---

## 🏁 How to Run

1. Clone the repo:
    ```bash
    git clone https://github.com/debjeetism/credit-card-fraud-detection-pipeline.git
    cd credit-card-fraud-detection-pipeline
    ```
2. Install dependencies (recommend in a virtual environment):
    ```bash
    pip install -r requirements.txt
    ```
   Or, manually install:
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost tensorflow keras
    ```

3. Download the [creditcard.csv](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) file from Kaggle and place in the repo root.

4. Open the notebook:
    ```
    jupyter notebook creditcardfraud_detection.ipynb
    ```
5. Run all cells.

---

## 📊 Results

- **Class imbalance crucially addressed:** Models evaluated with cross-validation and oversampling
- **Feature importance:** Analyzed top predictive features (due to PCA, variables labeled V1, V2, etc.)
- **Best test ROC-AUC:** Up to 0.985 with Random Forest and SMOTE

---

## 📎 References

- [Original Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Dal Pozzolo, A., et al., "Calibrating Probability with Undersampling for Unbalanced Classification", Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015.

---

## ⭐ Project Status

- Not productionized—educational/demo notebook for fraud detection techniques.
- Ready for extension with more advanced pipelines or integration into ML workflows.

---

## 📮 Contact

Questions? Raise an issue or connect via [GitHub](https://github.com/debjeetism).
