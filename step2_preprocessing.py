import pandas as pd

# Load the dataset we downloaded
df = pd.read_csv("titanic.csv")

print(" Before Cleaning:")
print(df.info())

# -------------------- CLEANING STEPS ---------------------

# 1) Drop columns that are useless for ML
# (These columns are purely text labels or duplicates)
df = df.drop(columns=["class", "alive"], errors="ignore")

# 2) Drop columns with too many missing values
# 'deck' has more than 70% missing
df = df.drop(columns=["deck"], errors="ignore")

# 3) Handle missing values
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])

# 4) Convert categorical text into numeric using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

print("\n After Cleaning & Encoding:")
print(df.info())

# 5) Save the cleaned dataset
df.to_csv("titanic_clean.csv", index=False)
print("\n Cleaned dataset saved as titanic_clean.csv")
