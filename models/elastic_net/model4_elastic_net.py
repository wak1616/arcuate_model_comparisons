#IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.utils import shuffle
from scipy.stats import uniform, loguniform
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

"""## Part 1: Getting Dataset into Pandas"""

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

# Set up features
features = [
'Age', 'Steep_axis_term', 'WTW_IOLMaster', 
  'MeanK_IOLMaster', 'Treated_astig', 'Type', 
  'AL', 'LASIK?'
]

# Identify numeric and categorical features
numeric_features = ['Age', 'Steep_axis_term', 'WTW_IOLMaster', 'MeanK_IOLMaster', 'Treated_astig', 'AL']
categorical_features = ['Type', 'LASIK?']

# Split dataset into training and testing sets
X = df[features]
y = df[target].values.ravel()  # Convert to 1D array for scikit-learn

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""## Part 2: Building an Elastic Net Regression Model with Hyperparameter Tuning"""

print("Setting up preprocessing pipeline...")
# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

# Define a function to run cross-validation and evaluate model
def evaluate_cv(model, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    mse_scores = []
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    
    for train_idx, test_idx in kf.split(X):
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        # Fit preprocessing and model
        model.fit(X_train_fold, y_train_fold)
        
        # Predict
        y_pred = model.predict(X_test_fold)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_fold, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_fold, y_pred)
        r2 = r2_score(y_test_fold, y_pred)
        
        mse_scores.append(mse)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        
        print(f"Fold {len(mse_scores)}: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    
    print("\nCross-Validation Results (Mean ± Std):")
    print(f"MSE: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
    print(f"RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    print(f"MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    print(f"R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    
    return np.mean(rmse_scores)

# Create a directory to save results
output_dir = Path(__file__).parent
os.makedirs(output_dir, exist_ok=True)

print("\n\n" + "="*50)
print("APPROACH 1: ELASTIC NET CV")
print("="*50)

# First approach: Use ElasticNetCV which performs internal cross-validation
print("\nRunning ElasticNetCV with internal cross-validation...")
elastic_net_cv = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
        alphas=np.logspace(-4, 1, 30),
        cv=5,
        max_iter=5000,
        tol=1e-3,
        random_state=42
    ))
])

# Fit the model
elastic_net_cv.fit(X_train, y_train)

# Get best parameters
en_cv_model = elastic_net_cv.named_steps['regressor']
best_alpha_cv = en_cv_model.alpha_
best_l1_ratio_cv = en_cv_model.l1_ratio_
print(f"Best alpha from ElasticNetCV: {best_alpha_cv:.6f}")
print(f"Best l1_ratio from ElasticNetCV: {best_l1_ratio_cv:.4f}")

# Evaluate on test set
y_pred_cv = elastic_net_cv.predict(X_test)
mse_cv = mean_squared_error(y_test, y_pred_cv)
rmse_cv = np.sqrt(mse_cv)
mae_cv = mean_absolute_error(y_test, y_pred_cv)
r2_cv = r2_score(y_test, y_pred_cv)

print("\nTest Set Metrics for ElasticNetCV:")
print(f"MSE: {mse_cv:.4f}")
print(f"RMSE: {rmse_cv:.4f}")
print(f"MAE: {mae_cv:.4f}")
print(f"R²: {r2_cv:.4f}")

print("\n\n" + "="*50)
print("APPROACH 2: RANDOMIZED SEARCH CV")
print("="*50)

# Second approach: Randomized search for even more extensive hyperparameter tuning
print("\nRunning RandomizedSearchCV for comprehensive hyperparameter tuning...")

# Create a pipeline with preprocessing and elastic net
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', ElasticNet(max_iter=5000, tol=1e-3, random_state=42))
])

# Set up parameter distributions for randomized search
param_distributions = {
    'regressor__alpha': loguniform(1e-5, 10),
    'regressor__l1_ratio': uniform(0, 1)
}

# Create randomized search
random_search = RandomizedSearchCV(
    pipe,
    param_distributions=param_distributions,
    n_iter=100,  # Number of parameter settings sampled
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# Fit the model
random_search.fit(X_train, y_train)

# Get best parameters
best_params = random_search.best_params_
best_alpha_rs = best_params['regressor__alpha']
best_l1_ratio_rs = best_params['regressor__l1_ratio']
print(f"Best parameters from RandomizedSearchCV: {best_params}")
print(f"Best score (negative MSE): {random_search.best_score_:.4f}")

# Evaluate on test set
best_model_rs = random_search.best_estimator_
y_pred_rs = best_model_rs.predict(X_test)
mse_rs = mean_squared_error(y_test, y_pred_rs)
rmse_rs = np.sqrt(mse_rs)
mae_rs = mean_absolute_error(y_test, y_pred_rs)
r2_rs = r2_score(y_test, y_pred_rs)

print("\nTest Set Metrics for Best Model from RandomizedSearchCV:")
print(f"MSE: {mse_rs:.4f}")
print(f"RMSE: {rmse_rs:.4f}")
print(f"MAE: {mae_rs:.4f}")
print(f"R²: {r2_rs:.4f}")

print("\n\n" + "="*50)
print("APPROACH 3: GRID SEARCH CV")
print("="*50)

# Third approach: Grid search for precise hyperparameter tuning around promising values
print("\nRunning GridSearchCV for precise hyperparameter tuning...")

# Create a focused grid search around the best values found so far
best_alpha = min(best_alpha_cv, best_alpha_rs)
best_l1_ratio = (best_l1_ratio_cv + best_l1_ratio_rs) / 2

# Set up parameter grid
alpha_grid = np.linspace(best_alpha * 0.1, best_alpha * 10, 20)
l1_ratio_grid = np.linspace(max(0.1, best_l1_ratio - 0.2), min(0.99, best_l1_ratio + 0.2), 10)

param_grid = {
    'regressor__alpha': alpha_grid,
    'regressor__l1_ratio': l1_ratio_grid
}

# Create grid search
grid_search = GridSearchCV(
    pipe,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

# Fit the model
grid_search.fit(X_train, y_train)

# Get best parameters
best_params_gs = grid_search.best_params_
print(f"Best parameters from GridSearchCV: {best_params_gs}")
print(f"Best score (negative MSE): {grid_search.best_score_:.4f}")

# Evaluate on test set
best_model_gs = grid_search.best_estimator_
y_pred_gs = best_model_gs.predict(X_test)
mse_gs = mean_squared_error(y_test, y_pred_gs)
rmse_gs = np.sqrt(mse_gs)
mae_gs = mean_absolute_error(y_test, y_pred_gs)
r2_gs = r2_score(y_test, y_pred_gs)

print("\nTest Set Metrics for Best Model from GridSearchCV:")
print(f"MSE: {mse_gs:.4f}")
print(f"RMSE: {rmse_gs:.4f}")
print(f"MAE: {mae_gs:.4f}")
print(f"R²: {r2_gs:.4f}")

# Select the best overall model
print("\n\n" + "="*50)
print("FINAL MODEL SELECTION")
print("="*50)

# Compare all three approaches
results = {
    'ElasticNetCV': (elastic_net_cv, mse_cv, rmse_cv, mae_cv, r2_cv),
    'RandomizedSearchCV': (best_model_rs, mse_rs, rmse_rs, mae_rs, r2_rs),
    'GridSearchCV': (best_model_gs, mse_gs, rmse_gs, mae_gs, r2_gs)
}

# Find the model with the lowest RMSE
best_approach = min(results.items(), key=lambda x: x[1][2])
best_approach_name = best_approach[0]
best_final_model = best_approach[1][0]
best_metrics = best_approach[1][1:]

print(f"\nThe best approach is: {best_approach_name}")
print(f"MSE: {best_metrics[0]:.4f}")
print(f"RMSE: {best_metrics[1]:.4f}")
print(f"MAE: {best_metrics[2]:.4f}")
print(f"R²: {best_metrics[3]:.4f}")

# Cross-validate the best model to get more reliable performance estimates
print("\nCross-validating the best model...")
final_cv_rmse = evaluate_cv(best_final_model, X, y)

# Save the best model
model_file = output_dir / 'elastic_net_best_model.pkl'
joblib.dump(best_final_model, model_file)
print(f"\nBest model saved to {model_file}")

# Feature importance analysis
print("\nAnalyzing feature importance...")
if best_approach_name == 'ElasticNetCV':
    # Get feature names from the pipeline
    feature_names = (
        numeric_features +
        list(best_final_model.named_steps['preprocessor']
             .named_transformers_['cat']
             .get_feature_names_out(categorical_features))
    )
    
    # Get coefficients
    coefficients = best_final_model.named_steps['regressor'].coef_
    
    # Create a DataFrame for feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })
    
    # Sort by absolute coefficient value
    feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
    feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
    
    print("\nFeature Importance (by coefficient magnitude):")
    print(feature_importance[['Feature', 'Coefficient']].to_string(index=False))
    
    # Save feature importance to file
    feature_importance_file = output_dir / 'elastic_net_feature_importance.csv'
    feature_importance.to_csv(feature_importance_file, index=False)
    print(f"Feature importance saved to {feature_importance_file}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
    plt.title('Elastic Net Coefficients')
    plt.xlabel('Coefficient')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(output_dir / 'elastic_net_coefficients.png')
    plt.close()

# Visualize predictions vs actual
y_pred_final = best_final_model.predict(X_test)

plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred_final, alpha=0.5)
plt.xlabel('Actual Arcuate Sweep')
plt.ylabel('Predicted Arcuate Sweep')
plt.title('Elastic Net Regression: Actual vs Predicted')
plt.plot([10, 55], [10, 55], 'r--', alpha=0.5)
plt.tight_layout()
plt.savefig(output_dir / 'elastic_net_predictions_vs_actual.png')
plt.close()

# Plot residuals
residuals = y_pred_final - y_test
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_final, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Elastic Net Regression: Residual Plot')
plt.tight_layout()
plt.savefig(output_dir / 'elastic_net_residuals.png')
plt.close()

# Create a histogram of residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.tight_layout()
plt.savefig(output_dir / 'elastic_net_residuals_distribution.png')
plt.close()

# Save the parameters of the best model to a text file
params_file = output_dir / 'elastic_net_best_params.txt'
with open(params_file, 'w') as f:
    f.write("Elastic Net Best Parameters\n")
    f.write("===========================\n\n")
    
    if best_approach_name == 'ElasticNetCV':
        model = best_final_model.named_steps['regressor']
        f.write(f"Alpha: {model.alpha_:.6f}\n")
        f.write(f"L1 Ratio: {model.l1_ratio_:.4f}\n")
    else:
        f.write(f"Alpha: {best_params_gs['regressor__alpha']:.6f}\n")
        f.write(f"L1 Ratio: {best_params_gs['regressor__l1_ratio']:.4f}\n")
    
    f.write("\nCross-Validation Performance:\n")
    f.write(f"MSE: {best_metrics[0]:.4f}\n")
    f.write(f"RMSE: {best_metrics[1]:.4f}\n")
    f.write(f"MAE: {best_metrics[2]:.4f}\n")
    f.write(f"R²: {best_metrics[3]:.4f}\n")

print(f"\nModel parameters saved to {params_file}")

print("\nElastic Net model training completed successfully!")
print("\nAll analysis and model files saved to the elastic_net directory.") 