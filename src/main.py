import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import shap
import joblib

# Load dataset (replace with your dataset)
df = pd.read_csv('data/data.csv')

# Example: Binary classification
target = 'target_column'
X = df.drop(columns=[target])
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Models to test
models = {
    'logreg': LogisticRegression(max_iter=1000),
    'svc': SVC(),
    'rf': RandomForestClassifier()
}

params = {
    'logreg': {'model__C': [0.1, 1, 10]},
    'svc': {'model__C': [0.1, 1, 10], 'model__kernel': ['linear', 'rbf']},
    'rf': {'model__n_estimators': [50, 100], 'model__max_depth': [None, 10]}
}

best_model = None
best_score = 0
best_name = ''
best_pipe = None

for name in models:
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', models[name])])
    grid = GridSearchCV(pipe, param_grid=params[name], cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    score = grid.best_score_
    if score > best_score:
        best_score = score
        best_model = grid.best_estimator_
        best_name = name
        best_pipe = grid

print(f"Best Model: {best_name} with score {best_score:.4f}")
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(best_model, 'model/best_model.pkl')

# SHAP explainability (works best with tree-based models)
explainer = shap.Explainer(best_model.named_steps['model'], best_model.named_steps['preprocessor'].transform(X_test))
shap_values = explainer(best_model.named_steps['preprocessor'].transform(X_test))
shap.summary_plot(shap_values, feature_names=best_model.named_steps['preprocessor'].get_feature_names_out())
