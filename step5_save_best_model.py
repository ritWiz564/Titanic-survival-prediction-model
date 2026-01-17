import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib

# ------------------ LOAD TRAIN DATA ------------------
X_train = pd.read_csv("X_train.csv").values
y_train = pd.read_csv("y_train.csv")["survived"].values

# ------------------ DEFINE MODELS ------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
}

# ------------------ LOAD MODEL COMPARISON ------------------
results_df = pd.read_csv("model_comparison.csv")
best_model_name = results_df.iloc[0]["Model"]
print(f"Best model based on ROC-AUC: {best_model_name}")

# ------------------ TRAIN BEST MODEL ------------------
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

# ------------------ SAVE MODEL ------------------
joblib.dump(best_model, "best_model_titanic.pkl")
print(" Best model saved as best_model_titanic.pkl")
