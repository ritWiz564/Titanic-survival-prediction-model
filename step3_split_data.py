import pandas as pd
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
df = pd.read_csv("titanic_clean.csv")

# Separate features (X) and target (y)
X = df.drop(columns=["survived"])
y = df["survived"]

# Perform train-test split
# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(" Training set size:", len(X_train))
print(" Testing set size:", len(X_test))

# Save these for the next steps
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\n Data split complete. Files created:")
print("X_train.csv, X_test.csv, y_train.csv, y_test.csv")
