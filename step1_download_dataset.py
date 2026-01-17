import seaborn as sns
import pandas as pd

# Load Titanic dataset directly from seaborn
df = sns.load_dataset("titanic")

# Save it as CSV for later steps
df.to_csv("titanic.csv", index=False)

print(" Titanic dataset downloaded and saved as titanic.csv")
print("Number of rows:", len(df))
print("Columns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
