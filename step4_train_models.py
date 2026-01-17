import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# ------------------ LOAD DATA ------------------
X_train = pd.read_csv("X_train.csv").values
X_test = pd.read_csv("X_test.csv").values
y_train = pd.read_csv("y_train.csv")["survived"].values
y_test = pd.read_csv("y_test.csv")["survived"].values

# ------------------ DEFINE MODELS ------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
}

results = []

plt.figure(figsize=(8,6))

# ------------------ TRAIN & EVALUATE ------------------
for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:,1]
    else:
        y_proba = model.decision_function(X_test)
    
    # Convert all to NumPy arrays to satisfy Pylance
    y_pred = np.array(y_pred)
    y_test_arr = np.array(y_test)
    y_proba = np.array(y_proba)
    
    # Evaluate
    acc = accuracy_score(y_test_arr, y_pred)
    prec = precision_score(y_test_arr, y_pred, zero_division=0)
    rec = recall_score(y_test_arr, y_pred, zero_division=0)
    f1 = f1_score(y_test_arr, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test_arr, y_proba)
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC-AUC": roc_auc
    })
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test_arr, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

# Random chance line
plt.plot([0,1],[0,1],'--', linewidth=0.7)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Titanic Models")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_all_models.png", dpi=150)
plt.show()

# ------------------ RESULTS TABLE ------------------
results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
print("\n Model Comparison:")
print(results_df)

# Save results
results_df.to_csv("model_comparison.csv", index=False)
print("\n model_comparison.csv saved")
