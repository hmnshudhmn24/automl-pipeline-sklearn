# 🤖 AutoML Pipeline with Scikit-learn

This project builds an **AutoML system** using **Scikit-learn** that automates:
data preprocessing, model selection, hyperparameter tuning, and model explainability using SHAP.

## 📊 Dataset

- Use your own CSV dataset by placing it in the `data/` folder as `data.csv`.
- Make sure to update the target column name in `main.py`.

## ⚙️ ML Pipeline

- **Preprocessing:** Handles numeric & categorical features with scaling and encoding.
- **Models Evaluated:**
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest Classifier
- **Tuning:** Uses `GridSearchCV` to find the best model and hyperparameters.

## 🧠 Explainability

- SHAP (SHapley Additive exPlanations) is used to visualize feature contributions.

## 📦 Requirements

```bash
pip install pandas numpy scikit-learn shap joblib matplotlib
```

## 🚀 How to Run

```bash
cd src
python main.py
```

- Outputs include model performance, SHAP summary plot, and saved model file (`best_model.pkl`).

## 📁 Files

- `src/main.py` — Main training, tuning, and explainability script.
- `data/data.csv` — Placeholder dataset (replace with your data).
- `README.md` — Project documentation.

## 🏁 Output

- Best model printed with evaluation metrics (accuracy, precision, recall, F1-score).
- SHAP summary plot saved/displayed for feature insights.
- Model saved in `model/` folder.

---

🔍 Made with ❤️ using Scikit-learn, SHAP, and Python.
