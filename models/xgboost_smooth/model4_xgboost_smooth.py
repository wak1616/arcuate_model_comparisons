#IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from scipy.interpolate import make_interp_spline
from sklearn.utils import shuffle

"""## Part 1: Getting Dataset into Pandas"""

# Set up output directory for saving files
current_dir = Path(__file__).parent
output_dir = current_dir
os.makedirs(output_dir, exist_ok=True)

#LOAD MAIN DATASET
datasets_dir = Path(__file__).resolve().parents[2] / "data"  # Project root directory + data folder
df = pd.read_csv(datasets_dir / "datafinal.csv", encoding = 'utf-8')

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
  'Treated_astig', 'Type', 
  'AL', 'LASIK?'
]

# Split dataset into training and testing sets
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


"""## Part 2: Building a smoothed XGBoost model with monotonicity constraints"""

# Define monotonicity constraints (1 for positive monotonic, -1 for negative, 0 for no constraint)
monotonic_constraints = {feature: 0 for feature in features}
# Add constraints for specified features based on correlation with target variable
monotonic_constraints['Age'] = 0  # No constraint
monotonic_constraints['Steep_axis_term'] = 0  # No constraint
monotonic_constraints['WTW_IOLMaster'] = 0  # No constraint
monotonic_constraints['Treated_astig'] = 1  # Strong positive correlation
monotonic_constraints['AL'] = 0  # No constraint

# Getting data ready as DMatrix for XGBoost:
dtrain = xgb.DMatrix(data=X_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(data=X_test, label=y_test, enable_categorical=True)
dfull = xgb.DMatrix(data=X, label=y, enable_categorical=True)

# Set fine-tuned hyperparameters for smoother predictions
# - Increased tree depth (6 instead of 4)
# - Reduced learning rate (0.02 instead of 0.05)
# - Increased number of trees (300 instead of 100)
# - Added min_child_weight to prevent overly specific splits
# - Increased subsample and colsample_bytree for better generalization
params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,              # Increased from 4 for finer gradations
    'learning_rate': 0.02,       # Reduced from 0.05 for smoother steps
    'subsample': 0.85,           # Slightly increased from 0.8
    'colsample_bytree': 0.9,     # Allows for more diverse trees
    'reg_alpha': 1.0,            
    'reg_lambda': 5.0,          
    'random_state': 42,
    'gamma': 0.1,               
    'min_child_weight': 3,       # Added to prevent overly specific splits
    'monotone_constraints': (0, 0, 0, 1, 0, 0, 0)  # Only Treated_astig has monotonic constraint
}

# Initialize/define the model by training it
print("Starting model training...")

model = xgb.train(
    params = params,
    dtrain = dtrain,
    num_boost_round = 300,       # Increased from previous models
    evals = [(dtrain, 'train'), (dtest, 'test')],
    verbose_eval = False,        
    early_stopping_rounds = 30   
)
print(f"Initial model trained with best iteration: {model.best_iteration}")

# Use XGBoost's built-in CV to figure out optimal number of boosting rounds to train full dataset
print("Starting cross-validation...")
cv_results = xgb.cv(
    params=params,
    dtrain=dfull,
    num_boost_round=300,
    early_stopping_rounds=20,
    nfold=5,
    metrics=['rmse'],
    seed=42
)

# Get the optimal number of rounds
ideal_boost_rounds = len(cv_results)
print(f"Optimal rounds from XGBoost CV: {ideal_boost_rounds}")

# Function to evaluate model with k-fold cross validation
def xgb_cv_score(X, y, params, n_splits=5):
    # Create KFold object
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Create empty lists to store scores for each fold
    mae_scores = []
    rmse_scores = []
    r2_scores = []

    # Loop through each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        # Get training data for this fold using indices
        X_train_fold = X.iloc[train_idx]  
        y_train_fold = y.iloc[train_idx]  

        # Get validation data for this fold using indices
        X_val_fold = X.iloc[val_idx]     
        y_val_fold = y.iloc[val_idx]      

        # Convert to XGBoost's DMatrix format
        dtrain_fold = xgb.DMatrix(X_train_fold, label=y_train_fold, enable_categorical=True)
        dval_fold = xgb.DMatrix(X_val_fold, label=y_val_fold, enable_categorical=True)

        # Train model on this fold
        model_fold = xgb.train(
            params=params,                   
            dtrain=dtrain_fold,              
            num_boost_round=300,             
            early_stopping_rounds=20,        
            evals=[(dtrain_fold, 'train'), (dval_fold, 'val')],
            verbose_eval=False               
        )

        # Make predictions on validation set
        y_pred_fold = model_fold.predict(dval_fold)

        # Calculate performance metrics for this fold
        mae = mean_absolute_error(y_val_fold, y_pred_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
        r2 = r2_score(y_val_fold, y_pred_fold)

        # Store scores for this fold
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        num_epochs = model_fold.best_iteration

        # Print results for this fold
        print(f"Fold {fold} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, Epochs: {num_epochs}")

    # After all folds are done, print average scores
    print("\nAverage scores across all folds:")
    print(f"MAE: {np.mean(mae_scores):.4f} (+/- {np.std(mae_scores):.4f})")
    print(f"RMSE: {np.mean(rmse_scores):.4f} (+/- {np.std(rmse_scores):.4f})")
    print(f"R2: {np.mean(r2_scores):.4f} (+/- {np.std(r2_scores):.4f})")
    print(f"Epochs: {np.mean(num_epochs):.2f} (+/- {np.std(num_epochs):.2f})")

    return mae_scores, rmse_scores, r2_scores

# Call the xgb_cv_score function with your data
print("Performing 5-fold cross-validation on full dataset:")
cv_mae, cv_rmse, cv_r2 = xgb_cv_score(X, y, params)

print("Training final model on full dataset...")
model_full = xgb.train(
    params = params,
    dtrain = dfull,
    num_boost_round = ideal_boost_rounds,
    evals = [(dfull, 'full_dataset')],
    verbose_eval = False,  
)
print("Final model training completed.")

# Predict on the test set and full dataset
print("Generating predictions...")
y_pred = model.predict(dtest)
y_pred_full = model_full.predict(dfull)
print("Predictions generated.")

# Create partial dependence plot for Treated_astig to visualize smoothness
print("Generating partial dependence plot for Treated_astig...")
# Create a sklearn-compatible wrapper for the XGBoost model
class XGBWrapper:
    def __init__(self, model):
        self.model = model
        
    def predict(self, X):
        dmatrix = xgb.DMatrix(X, enable_categorical=True)
        return self.model.predict(dmatrix)

# Create a feature grid just for Treated_astig to get higher resolution visualization
treated_astig_values = np.linspace(X['Treated_astig'].min(), X['Treated_astig'].max(), 100)

# Function to generate predictions for a range of Treated_astig values
def predict_for_astig_range(model, X_sample, astig_values):
    predictions = []
    for astig in astig_values:
        X_temp = X_sample.copy()
        X_temp['Treated_astig'] = astig
        dmatrix = xgb.DMatrix(X_temp, enable_categorical=True)
        pred = model.predict(dmatrix)
        predictions.append(pred.mean())
    return predictions

# Use a sample of records and vary only Treated_astig
X_sample = X.sample(10, random_state=42)
predictions = predict_for_astig_range(model_full, X_sample, treated_astig_values)

# Plot the relationship between Treated_astig and predicted Arcuate_sweep_total
plt.figure(figsize=(10, 6))
plt.plot(treated_astig_values, predictions, 'b-', linewidth=2)
plt.xlabel('Treated Astigmatism (D)')
plt.ylabel('Predicted Arcuate Sweep')
plt.title('Relationship Between Treated Astigmatism and Predicted Arcuate Sweep\n(Smoother XGBoost)')
plt.grid(True, alpha=0.3)
plt.savefig(output_dir / 'treated_astig_vs_arcuate_sweep_smooth.png')
plt.close()

# Visualize Predictions vs Actual Values
plt.figure(figsize=(10,10))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Arcuate Sweep')
plt.ylabel('Predicted Arcuate Sweep')
plt.title('Arcuate Sweep Prediction - Smoother XGBoost with Monotonic Constraints')
plt.plot([10, 55], [10, 55], 'r--', alpha=0.5)
plt.savefig(output_dir / 'model4_predictions_vs_actual.png')
plt.close()

# Calculating Performance Metrics (TEST DATASET)
mae_test = mean_absolute_error(y_test, y_pred)
mse_test = mean_squared_error(y_test, y_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_pred)

print("\nTEST DATASET METRICS:")
print(f"Mean Absolute Error: {mae_test:.4f}")
print(f"Mean Squared Error: {mse_test:.4f}")
print(f"Root Mean Squared Error: {rmse_test:.4f}")
print(f"R^2 Score: {r2_test:.4f}")

# Calculating Performance Metrics (FULL DATASET)
mae_full = mean_absolute_error(y, y_pred_full)
mse_full = mean_squared_error(y, y_pred_full)
rmse_full = np.sqrt(mse_full)
r2_full = r2_score(y, y_pred_full)

print("\nFULL DATASET METRICS:")
print(f"Mean Absolute Error: {mae_full:.4f}")
print(f"Mean Squared Error: {mse_full:.4f}")
print(f"Root Mean Squared Error: {rmse_full:.4f}")
print(f"R^2 Score: {r2_full:.4f}")

# Plot the residuals
residuals = y_pred_full.ravel() - y.to_numpy().ravel()
plt.hist(residuals, bins=50, edgecolor='k', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Arcuate Prediction Error (Smoother XGBoost)')
plt.savefig(output_dir / 'model4_residuals_distribution.png')
plt.close()

# Examine feature importance
plt.figure(figsize=(10, 8))
xgb.plot_importance(model_full)
plt.title('Feature Importance (Smoother XGBoost)')
plt.savefig(output_dir / 'model4_feature_importance.png')
plt.close()

# Print feature importance as text
importance = model_full.get_score(importance_type='weight')
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
print("\nFeature Importance (weight):")
for feature, score in sorted_importance:
    print(f"{feature}: {score}")

# Compare with previous model (model3)
# We'll generate a comparison plot to show how predictions vary with Treated_astig
try:
    # Try to load the previous model (model3)
    prev_model_path = Path(__file__).resolve().parents[1] / "xgboost_monotonic" / "XGBoost_monotonic_model_latest.json"
    if prev_model_path.exists():
        print("\nComparing with previous model (model3)...")
        prev_model = xgb.Booster()
        prev_model.load_model(str(prev_model_path))
        
        # Generate predictions for varying Treated_astig values
        prev_predictions = []
        for astig in treated_astig_values:
            X_temp = X_sample.copy()
            X_temp['Treated_astig'] = astig
            dmatrix = xgb.DMatrix(X_temp, enable_categorical=True)
            pred = prev_model.predict(dmatrix)
            prev_predictions.append(pred.mean())
        
        # Plot comparison
        plt.figure(figsize=(12, 7))
        plt.plot(treated_astig_values, predictions, 'b-', linewidth=2, label='Model 4 (Smoother)')
        plt.plot(treated_astig_values, prev_predictions, 'r--', linewidth=2, label='Model 3')
        plt.xlabel('Treated Astigmatism (D)')
        plt.ylabel('Predicted Arcuate Sweep')
        plt.title('Comparison of Model Predictions')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(output_dir / 'model_comparison_smoothness.png')
        plt.close()
        print(f"Comparison plot saved as '{output_dir / 'model_comparison_smoothness.png'}'")
    else:
        print("Previous model file not found for comparison.")
except Exception as e:
    print(f"Error comparing with previous model: {e}")

# Save the model
model_full.save_model(str(output_dir / 'XGBoost_smooth_model_latest.json'))
print(f"\nModel saved as '{output_dir / 'XGBoost_smooth_model_latest.json'}'")

# Create a summary table for quick reference
print("\n" + "="*60)
print("SMOOTHER ARCUATE SWEEP PREDICTION MODEL (MODEL 4)")
print("="*60)
print("\nThis model uses fine-tuned XGBoost with adjusted hyperparameters to provide:")
print("• Smoother predictions with gradual transitions")
print("• Maintained monotonicity on Treated_astig")
print("• Potentially improved accuracy with deeper trees and more careful regularization")
print("\nKey parameter adjustments:")
print("• Increased tree depth: 6 (vs. 4 in model3)")
print("• Reduced learning rate: 0.02 (vs. 0.05 in model3)")
print("• Added min_child_weight: 3 (not used in model3)")
print("• Increased number of trees")

print("\nModel Features (in order of importance):")
for i, (feature, score) in enumerate(sorted_importance, 1):
    print(f"{i}. {feature}")

print("\nModel Performance:")
print(f"R² Score: {r2_test:.4f} (test) / {r2_full:.4f} (full)")
print(f"RMSE: {rmse_test:.4f} (test) / {rmse_full:.4f} (full)")
print(f"MAE: {mae_test:.4f} (test) / {mae_full:.4f} (full)")

print(f"\nTo visualize the smoother predictions, see '{output_dir / 'treated_astig_vs_arcuate_sweep_smooth.png'}'")
print(f"To compare with the previous model, see '{output_dir / 'model_comparison_smoothness.png'}'") 