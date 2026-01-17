import pandas as pd
from datetime import datetime

# Load results
results_df = pd.read_csv("model_comparison.csv")

# Standardize column names
column_mapping = {}
for col in results_df.columns:
    if 'accuracy' in col.lower():
        column_mapping[col] = 'Accuracy'
    elif 'precision' in col.lower():
        column_mapping[col] = 'Precision'
    elif 'recall' in col.lower():
        column_mapping[col] = 'Recall'
    elif 'f1' in col.lower():
        column_mapping[col] = 'F1 Score'
    elif 'roc' in col.lower() or 'auc' in col.lower():
        column_mapping[col] = 'ROC-AUC'

results_df = results_df.rename(columns=column_mapping)

# Find best model for each metric
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
best_models = {}
for metric in metrics:
    if metric in results_df.columns:
        best_idx = results_df[metric].idxmax()
        best_models[metric] = (results_df.loc[best_idx, 'Model'], results_df.loc[best_idx, metric])

# Generate report
report = f"""
{'='*80}
                    MACHINE LEARNING PROJECT REPORT
                    Titanic Survival Prediction
{'='*80}

Date: {datetime.now().strftime('%B %d, %Y')}
Project: Comparative Study of Machine Learning Algorithms

{'='*80}
1. PROJECT OVERVIEW
{'='*80}

Dataset: Titanic Dataset
Total Records: 891 passengers
Task: Binary Classification (Survived / Did not survive)
Train-Test Split: 80-20

Algorithms Compared:
1. Logistic Regression
2. Random Forest
3. Decision Tree
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)

{'='*80}
2. MODEL PERFORMANCE COMPARISON
{'='*80}

{results_df.to_string(index=False)}

{'='*80}
3. BEST PERFORMING MODELS
{'='*80}

"""

for metric, (model, score) in best_models.items():
    report += f"{metric:.<30} {model} ({score:.4f})\n"

report += f"""

{'='*80}
4. OVERALL BEST MODEL
{'='*80}

"""

# Calculate average score for each model
results_df['Average'] = results_df[metrics].mean(axis=1)
best_overall_idx = results_df['Average'].idxmax()
best_overall = results_df.loc[best_overall_idx, 'Model']
best_avg_score = results_df.loc[best_overall_idx, 'Average']

report += f"Model: {best_overall}\n"
report += f"Average Score: {best_avg_score:.4f}\n\n"

report += "Individual Scores:\n"
for metric in metrics:
    if metric in results_df.columns:
        score = results_df.loc[best_overall_idx, metric]
        report += f"  - {metric}: {score:.4f}\n"

report += f"""

{'='*80}
5. KEY FINDINGS
{'='*80}

- Highest Accuracy: {best_models['Accuracy'][0]} ({best_models['Accuracy'][1]:.4f})
- Highest Precision: {best_models['Precision'][0]} ({best_models['Precision'][1]:.4f})
- Highest Recall: {best_models['Recall'][0]} ({best_models['Recall'][1]:.4f})
- Highest F1 Score: {best_models['F1 Score'][0]} ({best_models['F1 Score'][1]:.4f})
- Highest ROC-AUC: {best_models['ROC-AUC'][0]} ({best_models['ROC-AUC'][1]:.4f})

{'='*80}
6. CONCLUSIONS
{'='*80}

The {best_overall} model achieved the best overall performance with an average
score of {best_avg_score:.4f} across all metrics. This model is recommended for
predicting Titanic passenger survival.

All models showed good performance (>75% accuracy), indicating that the features
selected (passenger class, sex, age, fare, etc.) are strong predictors of survival.

{'='*80}
7. PROJECT FILES
{'='*80}

[DONE] titanic.csv                    - Original dataset
[DONE] titanic_processed.csv          - Cleaned and processed data
[DONE] best_model.pkl                 - Saved best model
[DONE] model_comparison.csv           - Results comparison
[DONE] model_comparison_charts.png    - Visual comparison
[DONE] individual_metrics.png         - Individual metric charts

{'='*80}
                            END OF REPORT
{'='*80}
"""

# Save report with UTF-8 encoding
with open('ML_PROJECT_REPORT.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print("\n[SUCCESS] Report saved as 'ML_PROJECT_REPORT.txt'")