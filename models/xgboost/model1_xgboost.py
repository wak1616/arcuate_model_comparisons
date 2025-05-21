#IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from scipy.interpolate import make_interp_spline
from sklearn.metrics import make_scorer
from sklearn.utils import shuffle

"""## Part 1: Getting Dataset into Pandas"""

#LOAD MAIN DATASET
datasets_dir = Path(__file__).resolve().parents[2] / "data"  # Project root directory + data folder
df = pd.read_csv(datasets_dir / "datafinal.csv", encoding = 'utf-8')

# SET RANDOM_STATE AND SHUFFLE THE DATASET ('df')
df = shuffle(df, random_state=42)

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


"""## Part 2: Building a model that predicts Arcuate Sweep"""

# Getting data ready as DMatrix for XGBoost:
dtrain = xgb.DMatrix(data=X_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(data=X_test, label=y_test, enable_categorical=True)
dfull = xgb.DMatrix(data=X, label=y, enable_categorical=True)

# Set hyperparameters
params = {
    'objective': 'reg:squarederror',
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 1.0,
    'reg_alpha': 1.0,
    'reg_lambda': 5.0,
    'random_state': 42,
    'gamma': 0.1
    }


# Note: These optimized parameters and features are based on: gamma = 0.1
# The model was improved through 
# grid search and provide better generalization performance and more balanced
# feature importance distribution.

# Initialize/define the model by training it
print("Starting model training...")

model = xgb.train(
    params = params,
    dtrain = dtrain,
    num_boost_round = 1000, #specify number of boosting rounds
    evals = [(dtrain, 'train'), (dtest, 'test')],
    verbose_eval = False,  # Changed from 10 to False to reduce output
    early_stopping_rounds = 30 #stop if no improvement after XX rounds
)
print(f"Initial model trained with best iteration: {model.best_iteration}")

# Use XGBoost's built-in CV to to figure out optimal number of boosting rounds to train full dataset
print("Starting cross-validation...")
best_rounds = xgb.cv(
    params=params,
    dtrain=dfull,
    num_boost_round=1000,
    early_stopping_rounds=20,
    nfold=5,
    metrics=['rmse'],
    seed=42
)

# output of "best_rounds" is actually a pandas df that will show rows with index of boosting round # followed by accuracy metric. each row is the average of all 5 models.
# unlike sklearn's KFold, XGBoost's CV only only shows averages of all 5 folds (not individual fold data)
# XGBoost's CV is perfect for determining optimal number of rounds.

# Get the optimal number of rounds
ideal_boost_rounds = len(best_rounds)
print(f"Optimal rounds from XGBoost CV: {ideal_boost_rounds}")

def xgb_cv_score(X, y, params, n_splits=5):
    # Create KFold object: will split data into 5 parts
    # shuffle=True means data will be randomly shuffled before splitting
    # random_state=42 ensures we get same random splits each time we run
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Create empty lists to store scores for each fold
    mae_scores = []
    rmse_scores = []
    r2_scores = []

    # Loop through each fold
    # kf.split(X) creates indices for train/validation split for each fold
    # enumerate(kf.split(X), 1) starts counting folds from 1 instead of 0
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        # Get training data for this fold using indices
        X_train_fold = X.iloc[train_idx]  # Features for training
        y_train_fold = y.iloc[train_idx]  # Target for training

        # Get validation data for this fold using indices
        X_val_fold = X.iloc[val_idx]      # Features for validation
        y_val_fold = y.iloc[val_idx]      # Target for validation

        # Convert to XGBoost's DMatrix format
        dtrain_fold = xgb.DMatrix(X_train_fold, label=y_train_fold, enable_categorical=True)
        dval_fold = xgb.DMatrix(X_val_fold, label=y_val_fold, enable_categorical=True)

        # Train model on this fold
        model_fold = xgb.train(
            params=params,                     # Hyperparameters
            dtrain=dtrain_fold,               # Training data
            num_boost_round=1000,# Number of trees
            early_stopping_rounds=20,         # Stop if no improvement after 20 rounds
            evals=[(dtrain_fold, 'train'), (dval_fold, 'val')],  # Evaluation sets
            verbose_eval=False                 # Don't print progress
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
    print(f"Epochs: {np.mean(num_epochs)} (+/- {np.std(num_epochs)})")

    return mae_scores, rmse_scores, r2_scores

# Call the xgb_cv_score function with your data
print("Performing 5-fold cross-validation on full dataset:")
cv_mae, cv_rmse, cv_r2 = xgb_cv_score(X, y, params)

print("Training final model on full dataset...")
model_full = xgb.train(
    params = params,
    dtrain = dfull,
    num_boost_round = ideal_boost_rounds, #specify number of boosting rounds
    evals = [(dfull, 'full_dataset')],
    verbose_eval = False,  # Changed from 10 to False
)
print("Final model training completed.")

# Predict on the test set (which will output an np.array)
print("Generating predictions...")
y_pred = model.predict(dtest)
y_pred_full = model_full.predict(dfull)
print("Predictions generated.")

# Visualize Predictions vs Actual Values for full resdiual model (TEST DATASET)
plt.figure(figsize=(10,10))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Acruate Sweep')
plt.ylabel('Predicted Arcuate Sweep')
plt.title('Arcuate Sweep Prediction Model with Age × Treated_astig Interaction')
plt.plot([10, 55], [10, 55], 'r--', alpha=0.5)
plt.savefig('model_predictions_vs_actual.png')
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
plt.title('Distribution of Arcuate Prediction Error')
plt.savefig('residuals_distribution.png')
plt.close()

# Examine feature importance
plt.figure(figsize=(10, 8))
xgb.plot_importance(model_full)
plt.title('Feature Importance with Age × Treated_astig Interaction')
plt.savefig('feature_importance.png')
plt.close()

# Print feature importance as text
importance = model_full.get_score(importance_type='weight')
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
print("\nFeature Importance (weight):")
for feature, score in sorted_importance:
    print(f"{feature}: {score}")

# Save the model
model_full.save_model('XGBoost_model_latest.json')
print("\nModel saved as 'XGBoost_model_latest.json'")

# Create a summary table for quick reference
print("\n" + "="*60)
print("ARCUATE SWEEP PREDICTION MODEL - SUMMARY")
print("="*60)
print("\nModel Features (in order of importance):")
for i, (feature, score) in enumerate(sorted_importance, 1):
    print(f"{i}. {feature}")

print("\nModel Performance:")
print(f"R² Score: {r2_test:.4f} (test) / {r2_full:.4f} (full)")
print(f"RMSE: {rmse_test:.4f} (test) / {rmse_full:.4f} (full)")
print(f"MAE: {mae_test:.4f} (test) / {mae_full:.4f} (full)")

print("\nModel trained successfully with Age × Treated_astig interaction term.")
