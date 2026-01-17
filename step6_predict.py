import pandas as pd
import joblib

# ------------------ LOAD SAVED MODEL ------------------
model = joblib.load("best_model_titanic.pkl")
print(" Loaded best model!")

# ------------------ LOAD TRAINING COLUMNS ------------------
X_train = pd.read_csv("X_train.csv")
feature_cols = X_train.columns.tolist()
print(f" Model expects {len(feature_cols)} features.")

# ------------------ MANUAL INPUT FOR NEW PASSENGER ------------------
# Create a dictionary to fill new passenger info
print("\nEnter new passenger info:")

new_data = {}
for col in feature_cols:
    val = input(f"{col}: ")
    if val == "":
        val = 0  # fill missing with 0
    try:
        new_data[col] = float(val)
    except ValueError:
        new_data[col] = val

# Convert to DataFrame
new_passengers = pd.DataFrame([new_data], columns=feature_cols)

# ------------------ PREDICT ------------------
X_new = new_passengers.values
pred = model.predict(X_new)[0]
prob = model.predict_proba(X_new)[0,1]

# ------------------ SHOW RESULT ------------------
print(f"\n Prediction: {'Survived ' if pred==1 else 'Did not survive '}")
print(f" Probability of survival: {prob:.2f}")
