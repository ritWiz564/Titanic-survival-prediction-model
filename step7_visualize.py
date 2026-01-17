import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
results_df = pd.read_csv("model_comparison.csv")
print("Available columns:", results_df.columns.tolist())
print("\nResults:\n", results_df)

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))

# Check actual column names and standardize them
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
print("\nStandardized columns:", results_df.columns.tolist())

# Define metrics to plot
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
available_metrics = [m for m in metrics if m in results_df.columns]

# Convert to lists for type safety
model_names = results_df['Model'].tolist()

# 1. Grouped Bar Chart - All Metrics
plt.subplot(2, 2, 1)
x_pos = range(len(results_df))
width = 0.15
for i, metric in enumerate(available_metrics):
    plt.bar([p + width * i for p in x_pos], results_df[metric].values, 
            width=width, alpha=0.8, label=metric)
plt.xlabel('Models', fontsize=12, fontweight='bold')
plt.ylabel('Score', fontsize=12, fontweight='bold')
plt.title('Model Performance Comparison - All Metrics', fontsize=14, fontweight='bold')
plt.xticks([p + width * 2 for p in x_pos], model_names, rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', alpha=0.3)

# 2. Heatmap
plt.subplot(2, 2, 2)
heatmap_data = results_df[available_metrics].T
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGnBu', 
            xticklabels=model_names, yticklabels=available_metrics,
            cbar_kws={'label': 'Score'})
plt.title('Performance Heatmap', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')

# 3. Individual Metric Comparison - Accuracy
plt.subplot(2, 2, 3)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
bars = plt.bar(model_names, results_df['Accuracy'].values, color=colors, alpha=0.7)
plt.xlabel('Models', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy Score', fontsize=12, fontweight='bold')
plt.title('Accuracy Comparison', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylim([0.7, 0.85])
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# 4. ROC-AUC Comparison
plt.subplot(2, 2, 4)
bars = plt.bar(model_names, results_df['ROC-AUC'].values, color=colors, alpha=0.7)
plt.xlabel('Models', fontsize=12, fontweight='bold')
plt.ylabel('ROC-AUC Score', fontsize=12, fontweight='bold')
plt.title('ROC-AUC Comparison', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylim([0.7, 0.9])
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison_charts.png', dpi=300, bbox_inches='tight')
print("\n Visualization saved as 'model_comparison_charts.png'")
plt.show()

# Additional: Individual metric bar charts
fig, axes = plt.subplots(1, len(available_metrics), figsize=(20, 4))
for idx, metric in enumerate(available_metrics):
    axes[idx].bar(model_names, results_df[metric].values, color=colors, alpha=0.7)
    axes[idx].set_title(f'{metric}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Models', fontsize=10)
    axes[idx].set_ylabel('Score', fontsize=10)
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].grid(axis='y', alpha=0.3)
    # Add values on bars
    for i, v in enumerate(results_df[metric].values):
        axes[idx].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('individual_metrics.png', dpi=300, bbox_inches='tight')
print(" Individual metrics saved as 'individual_metrics.png'")
plt.show()