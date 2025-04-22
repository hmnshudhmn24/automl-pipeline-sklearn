# AutoML System with Scikit-learn Pipelines

This project is an automated machine learning (AutoML) system using Scikit-learn. It handles data preprocessing, model selection, hyperparameter tuning, and includes explainability using SHAP.

## Features

- Preprocessing with Pipelines (scaling, encoding)
- Model selection: Logistic Regression, SVM, Random Forest
- Hyperparameter tuning using GridSearchCV
- SHAP for model explainability
- Exports best model as a `.pkl` file

## How to Run

1. Place your dataset in the `data/` folder as `data.csv`
2. Update `target = 'target_column'` in `src/main.py` to match your dataset
3. Run the script:

```bash
cd src
python main.py
```

4. Check outputs, classification report, and SHAP plots.

## Requirements

- pandas, numpy, scikit-learn, shap, joblib, matplotlib
