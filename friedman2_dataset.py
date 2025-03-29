import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_friedman2_dataset(n_samples=1000, noise=0.1, random_state=42):
    """
    Generate the Friedman #2 synthetic regression dataset with the formula:
    y = (x1² + (x2·x3 - 1/(x2·x4))²)^0.5 + noise
    
    Features ranges:
    - 0 ≤ x1 ≤ 100
    - 40π ≤ x2 ≤ 560π 
    - 0 ≤ x3 ≤ 1
    - 1 ≤ x4 ≤ 11
    """
    np.random.seed(random_state)
    
    # Generate features within specified ranges
    x1 = np.random.uniform(0, 100, n_samples)
    x2 = np.random.uniform(40 * np.pi, 560 * np.pi, n_samples)
    x3 = np.random.uniform(0, 1, n_samples)
    x4 = np.random.uniform(1, 11, n_samples)
    
    # Create feature matrix
    X = np.column_stack((x1, x2, x3, x4))
    
    # Calculate target using the formula
    term1 = x1**2
    term2 = (x2 * x3 - 1 / (x2 * x4))**2
    y_without_noise = np.sqrt(term1 + term2)
    
    # Add Gaussian noise
    y = y_without_noise + np.random.normal(0, noise, n_samples)
    
    return X, y


# Create output directory
output_dir = "friedman2_output"
os.makedirs(output_dir, exist_ok=True)

# Generate dataset
n_samples = 2000
noise_level = 0.1
X, y = generate_friedman2_dataset(n_samples=n_samples, noise=noise_level)

# Create DataFrame
feature_names = ['x1', 'x2', 'x3', 'x4']
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Print dataset overview
print("Friedman #2 Dataset Overview:")
print(f"Number of samples: {n_samples}")
print(f"Noise level: {noise_level}")
print("\nDataset statistics:")
print(df.describe())

# Save raw dataset
df.to_csv(os.path.join(output_dir, "friedman2_raw.csv"), index=False)

# Visualize feature distributions
plt.figure(figsize=(15, 10))
for i, feature in enumerate(feature_names):
    plt.subplot(3, 2, i+1)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')

# Target distribution
plt.subplot(3, 2, 5)
sns.histplot(df['target'], kde=True)
plt.title('Distribution of target')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_distributions.png"))
plt.close()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.close()

# Feature vs target scatter plots
plt.figure(figsize=(15, 10))
for i, feature in enumerate(feature_names):
    plt.subplot(2, 2, i+1)
    sns.scatterplot(x=feature, y='target', data=df, alpha=0.6)
    plt.title(f'{feature} vs target')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_target_scatter.png"))
plt.close()

# Pairplot for feature relationships
plt.figure(figsize=(10, 8))
sns.pairplot(df.sample(500), height=2.5)  # Sample to speed up pairplot
plt.savefig(os.path.join(output_dir, "pairplot.png"))
plt.close()

# Prepare data for RoBERTa (convert numerical to text)
def convert_to_text_format(row):
    """Convert numerical features to text format for use with RoBERTa"""
    text = f"x1: {row['x1']:.2f}, x2: {row['x2']:.2f}, x3: {row['x3']:.4f}, x4: {row['x4']:.2f}"
    return text

# Add text representation column
df['text_features'] = df.apply(convert_to_text_format, axis=1)

# Display text format examples
print("\nExamples of text format for RoBERTa:")
for i in range(5):
    print(f"Sample {i+1}: {df['text_features'].iloc[i]}")
    print(f"Target: {df['target'].iloc[i]:.4f}")
    print()

# Split data (70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(
    df.drop('target', axis=1), df['target'], 
    test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Create dataframes for each split
train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

print(f"Train set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

# Save split datasets in CSV format
train_df.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
val_df.to_csv(os.path.join(output_dir, "validation_data.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test_data.csv"), index=False)

# Save datasets for RoBERTa (text format)
def save_for_roberta(df, filename):
    with open(filename, 'w') as f:
        for i, row in df.iterrows():
            f.write(f"{row['text_features']}\t{row['target']:.6f}\n")

save_for_roberta(train_df, os.path.join(output_dir, "train_roberta.txt"))
save_for_roberta(val_df, os.path.join(output_dir, "validation_roberta.txt"))
save_for_roberta(test_df, os.path.join(output_dir, "test_roberta.txt"))

print("\nDatasets saved in both CSV and RoBERTa-friendly formats.")

# Feature importance analysis
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train_df[feature_names], train_df['target'])

# Plot feature importances
plt.figure(figsize=(10, 6))
importances = model.feature_importances_
indices = np.argsort(importances)

plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Feature Importance')
plt.title('Feature Importance Analysis')
plt.savefig(os.path.join(output_dir, "feature_importance.png"))
plt.close()

print("\nFriedman #2 dataset exploration and preparation complete!") 