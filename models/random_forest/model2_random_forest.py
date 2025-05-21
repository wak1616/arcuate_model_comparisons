import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import joblib
import warnings
warnings.filterwarnings('ignore')

"""## Part 1: Getting Dataset into Pandas"""

#LOAD MAIN DATASET
# Get absolute path to the data file
current_file = Path(__file__)
project_root = current_file.parent.parent.parent  # Go up two levels to the project root
data_file = project_root / "data" / "datafinal.csv"

df = pd.read_csv(data_file, encoding = 'utf-8')

# SET RANDOM_STATE AND SHUFFLE THE DATASET ('df')
df = shuffle(df, random_state=42)

# find out how many entries are nan and in which columns
print(df.isna().sum())

#Remove trailing white spaces for 'Type', 'Sex', 'Eye', 'LASIK?'
df['Type'] = df['Type'].str.strip()
df['Sex'] = df['Sex'].str.strip()
df['Eye'] = df['Eye'].str.strip()
df['LASIK?'] = df['LASIK?'].str.strip()

# Replace any 'Type' entries that = "singe" to "single"
df['Type'] = df['Type'].replace('singe', 'single')

#specify the categorical variables
df['Type'] = df['Type'].astype('category')
df['Sex'] = df['Sex'].astype('category')
df['Eye'] = df['Eye'].astype('category')
df['LASIK?'] = df['LASIK?'].astype('category')

#Look for any outliers and get an overview of dataset
print(df.describe())
print(df.info())

# Setting up features and target.
target = ['Arcuate_sweep_total']

# Add the interaction term to the features list
features = [
'Age', 'Steep_axis_term', 'WTW_IOLMaster', 
  'MeanK_IOLMaster', 'Treated_astig', 'Type', 
  'AL', 'LASIK?'
]

# Split dataset into training and testing sets
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

"""## Part 2: Preprocessing for Random Forest"""

# Create a copy of the training and testing sets
X_train_rf = X_train.copy()
X_test_rf = X_test.copy()

# One-hot encode categorical features
categorical_features = ['Type', 'LASIK?']
encoder = OneHotEncoder(sparse_output=False, drop='first')

# Apply one-hot encoding to categorical features
cat_train = encoder.fit_transform(X_train_rf[categorical_features])
cat_test = encoder.transform(X_test_rf[categorical_features])

# Get feature names after one-hot encoding
cat_feature_names = []
for i, feature in enumerate(categorical_features):
    categories = list(encoder.categories_[i])[1:]  # Drop first category
    for category in categories:
        cat_feature_names.append(f"{feature}_{category}")

# Drop original categorical columns
X_train_rf = X_train_rf.drop(categorical_features, axis=1)
X_test_rf = X_test_rf.drop(categorical_features, axis=1)

# Convert to numpy arrays
X_train_rf_numeric = X_train_rf.values
X_test_rf_numeric = X_test_rf.values

# Combine numeric and one-hot encoded features
X_train_rf_final = np.hstack((X_train_rf_numeric, cat_train))
X_test_rf_final = np.hstack((X_test_rf_numeric, cat_test))

# Combined feature names
numeric_feature_names = X_train_rf.columns.tolist()
all_feature_names = numeric_feature_names + cat_feature_names

"""## Part 3: Building Random Forest Model with Optimal Hyperparameters"""

# Initialize and train the Random Forest model with optimal hyperparameters from rf_best_params.txt
best_rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42
)

# Train the model
print("\nTraining Random Forest model with optimal hyperparameters...")
best_rf.fit(X_train_rf_final, y_train.values.ravel())

# Make predictions with the model
y_pred = best_rf.predict(X_test_rf_final)

"""## Part 4: Model Evaluation and Results"""

# Evaluate model
print("\nRandom Forest Performance:")
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Get feature importances
feature_importances = best_rf.feature_importances_
importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)

print("\nFeature Importances:")
print(importance_df.head(10))

# Permutation importance for more reliable feature importance
perm_importance = permutation_importance(
    best_rf, X_test_rf_final, y_test.values.ravel(), 
    n_repeats=10, random_state=42
)

perm_importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': perm_importance.importances_mean
}).sort_values('Importance', ascending=False)

print("\nPermutation-based Feature Importances:")
print(perm_importance_df.head(10))

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
plt.title('Random Forest Feature Importances')
plt.tight_layout()
plt.savefig('feature_importance_rf.png')

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Random Forest: Actual vs Predicted Values')
plt.tight_layout()
plt.savefig('model_predictions_vs_actual_rf.png')

# Plot residuals
residuals = y_test.values.ravel() - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Random Forest: Residuals Distribution')
plt.tight_layout()
plt.savefig('residuals_distribution_rf.png')

# Get current directory
current_dir = Path(__file__).parent

# Save model results to file
with open(current_dir / 'rf_model_results.txt', 'w') as f:
    f.write("Random Forest Regression Model Results\n")
    f.write("======================================\n\n")
    f.write("Model Parameters:\n")
    f.write(f"n_estimators: 100\n")
    f.write(f"max_depth: 10\n")
    f.write(f"min_samples_split: 10\n")
    f.write(f"min_samples_leaf: 4\n")
    f.write(f"max_features: sqrt\n\n")
    f.write("Model Performance:\n")
    f.write(f"Mean Squared Error: {mse:.4f}\n")
    f.write(f"Root Mean Squared Error: {rmse:.4f}\n")
    f.write(f"Mean Absolute Error: {mae:.4f}\n")
    f.write(f"R² Score: {r2:.4f}\n\n")
    f.write("Top 10 Feature Importances:\n")
    for _, row in importance_df.head(10).iterrows():
        f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")

# Save the model
joblib.dump(best_rf, current_dir / 'random_forest_model.joblib')
print("\nRandom Forest model, results, and visualizations saved.") 