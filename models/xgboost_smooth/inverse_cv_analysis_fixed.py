"""
Inverse Cross-Validation Analysis for Treated_astig Prediction - FIXED VERSION

This script performs cross-validation analysis to determine error metrics (RMSE, R², etc.) 
for each fold when predicting "Treated_astig" values using the XGBoost model that was 
trained to predict "Arcuate_sweep_total". 

FIXES:
- Uses realistic Treated_astig bounds from training data (0.06 - 1.61)
- Sets tolerance to 0.01 (practical measurement precision)
- Better error handling and categorical feature management
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.utils import shuffle
from pathlib import Path
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize_scalar

# Set up paths
current_dir = Path(__file__).parent
output_dir = current_dir
datasets_dir = Path(__file__).resolve().parents[2] / "data"
os.makedirs(output_dir, exist_ok=True)

def load_and_prepare_data():
    """Load and prepare the dataset for analysis."""
    df = pd.read_csv(datasets_dir / "datafinal.csv", encoding='utf-8')
    df = shuffle(df, random_state=42)
    
    # Clean data
    df['Type'] = df['Type'].str.strip()
    df['Sex'] = df['Sex'].str.strip()
    df['Eye'] = df['Eye'].str.strip()
    df['LASIK?'] = df['LASIK?'].str.strip()
    df['Type'] = df['Type'].replace('singe', 'single')
    
    # Set categorical variables
    df['Type'] = df['Type'].astype('category')
    df['Sex'] = df['Sex'].astype('category')
    df['Eye'] = df['Eye'].astype('category')
    df['LASIK?'] = df['LASIK?'].astype('category')
    
    return df

def predict_arcuate_for_astig(model, patient_base_features, treated_astig_value):
    """
    Predict Arcuate_sweep_total for a patient with a given Treated_astig value.
    
    Args:
        model: Trained XGBoost model
        patient_base_features: Series with patient's features (without Treated_astig)
        treated_astig_value: The Treated_astig value to test
    
    Returns:
        Predicted Arcuate_sweep_total value
    """
    # Create complete feature set
    features_complete = patient_base_features.copy()
    features_complete['Treated_astig'] = treated_astig_value
    
    # Ensure proper feature order
    feature_order = ['Age', 'Steep_axis_term', 'WTW_IOLMaster', 'Treated_astig', 'Type', 'AL', 'LASIK?']
    ordered_features = features_complete[feature_order]
    
    # Create DataFrame for prediction
    pred_df = pd.DataFrame([ordered_features])
    
    # Set categorical types
    categorical_cols = ['Type', 'Sex', 'Eye', 'LASIK?']
    for col in categorical_cols:
        if col in pred_df.columns:
            pred_df[col] = pred_df[col].astype('category')
    
    # Make prediction
    dmatrix = xgb.DMatrix(pred_df, enable_categorical=True)
    return model.predict(dmatrix)[0]

def inverse_predict_treated_astig(model, patient_base_features, target_arcuate, 
                                astig_min, astig_max, tolerance=0.01):
    """
    Find the Treated_astig value that produces the target Arcuate_sweep_total.
    
    Args:
        model: Trained XGBoost model
        patient_base_features: Series with patient's features (without Treated_astig)
        target_arcuate: Target Arcuate_sweep_total value
        astig_min: Minimum possible Treated_astig value (from training data)
        astig_max: Maximum possible Treated_astig value (from training data)
        tolerance: Convergence tolerance (0.01 for practical precision)
    
    Returns:
        Predicted Treated_astig value
    """
    def objective(astig_val):
        try:
            predicted_arcuate = predict_arcuate_for_astig(model, patient_base_features, astig_val)
            return abs(predicted_arcuate - target_arcuate)
        except Exception:
            return 1000  # Large penalty for failed predictions
    
    # Use scipy's minimize_scalar for robust optimization
    result = minimize_scalar(objective, bounds=(astig_min, astig_max), method='bounded')
    
    if result.success and result.fun < 1000:
        return result.x
    else:
        # Fallback to binary search if optimization fails
        return binary_search_astig(model, patient_base_features, target_arcuate, astig_min, astig_max, tolerance)

def binary_search_astig(model, patient_base_features, target_arcuate, astig_min, astig_max, tolerance):
    """
    Binary search fallback for finding Treated_astig value.
    """
    max_iterations = 100
    iteration = 0
    
    while abs(astig_max - astig_min) > tolerance and iteration < max_iterations:
        astig_mid = (astig_min + astig_max) / 2
        try:
            predicted_arcuate = predict_arcuate_for_astig(model, patient_base_features, astig_mid)
            
            if predicted_arcuate < target_arcuate:
                astig_min = astig_mid
            else:
                astig_max = astig_mid
        except Exception:
            # If prediction fails, return the midpoint
            break
            
        iteration += 1
    
    return (astig_min + astig_max) / 2

def perform_inverse_cv_analysis(X, y, params, n_splits=5):
    """
    Perform cross-validation analysis for Treated_astig prediction.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store results for each fold
    fold_results = {
        'mse_scores': [],
        'rmse_scores': [],
        'mae_scores': [],
        'r2_scores': [],
        'explained_variance_scores': [],
        'fold_predictions': [],
        'fold_actuals': [],
        'fold_indices': []
    }
    
    print(f"Starting {n_splits}-fold cross-validation for Treated_astig prediction...")
    print(f"Dataset Treated_astig range: [{X['Treated_astig'].min():.3f}, {X['Treated_astig'].max():.3f}]")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\nProcessing Fold {fold}...")
        
        # Split data
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Get realistic bounds from training data for this fold
        fold_astig_min = max(X_train_fold['Treated_astig'].min(), 0.05)  # Minimum realistic value
        fold_astig_max = min(X_train_fold['Treated_astig'].max(), 2.0)   # Maximum realistic value
        
        print(f"  Fold {fold} Treated_astig bounds: [{fold_astig_min:.3f}, {fold_astig_max:.3f}]")
        
        # Train model on this fold
        dtrain_fold = xgb.DMatrix(X_train_fold, label=y_train_fold, enable_categorical=True)
        dval_fold = xgb.DMatrix(X_val_fold, label=y_val_fold, enable_categorical=True)
        
        model_fold = xgb.train(
            params=params,
            dtrain=dtrain_fold,
            num_boost_round=300,
            early_stopping_rounds=20,
            evals=[(dtrain_fold, 'train'), (dval_fold, 'val')],
            verbose_eval=False
        )
        
        # Perform inverse prediction for validation set
        predicted_astig_values = []
        actual_astig_values = X_val_fold['Treated_astig'].values
        
        print(f"  Performing inverse prediction for {len(val_idx)} samples...")
        
        for i, idx in enumerate(val_idx):
            if i % 25 == 0:  # Progress indicator
                print(f"    Processing sample {i+1}/{len(val_idx)}")
            
            # Get patient data
            patient_row = X_val_fold.iloc[i]
            actual_arcuate = y_val_fold.iloc[i]
            
            # Remove Treated_astig from patient features
            patient_base = patient_row.drop('Treated_astig')
            
            # Find Treated_astig value that would produce the actual Arcuate_sweep_total
            try:
                predicted_astig = inverse_predict_treated_astig(
                    model_fold, patient_base, actual_arcuate, 
                    astig_min=fold_astig_min, astig_max=fold_astig_max
                )
                predicted_astig_values.append(predicted_astig)
            except Exception as e:
                print(f"    Warning: Failed to predict for sample {i}: {e}")
                # Use fold median as fallback
                predicted_astig_values.append(np.median(X_train_fold['Treated_astig']))
        
        predicted_astig_values = np.array(predicted_astig_values)
        
        # Calculate metrics for this fold
        mse = mean_squared_error(actual_astig_values, predicted_astig_values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_astig_values, predicted_astig_values)
        r2 = r2_score(actual_astig_values, predicted_astig_values)
        explained_var = explained_variance_score(actual_astig_values, predicted_astig_values)
        
        # Store results
        fold_results['mse_scores'].append(mse)
        fold_results['rmse_scores'].append(rmse)
        fold_results['mae_scores'].append(mae)
        fold_results['r2_scores'].append(r2)
        fold_results['explained_variance_scores'].append(explained_var)
        fold_results['fold_predictions'].append(predicted_astig_values)
        fold_results['fold_actuals'].append(actual_astig_values)
        fold_results['fold_indices'].append(val_idx)
        
        print(f"  Fold {fold} Results:")
        print(f"    MSE: {mse:.4f}")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    MAE: {mae:.4f}")
        print(f"    R²: {r2:.4f}")
        print(f"    Explained Variance: {explained_var:.4f}")
    
    return fold_results

def plot_fold_results(fold_results, output_dir):
    """Create plots showing the results of the inverse CV analysis."""
    
    # Plot 1: Metrics across folds
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    folds = range(1, len(fold_results['mae_scores']) + 1)
    
    ax1.bar(folds, fold_results['mse_scores'])
    ax1.set_title('MSE across Folds')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('MSE')
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(folds, fold_results['rmse_scores'])
    ax2.set_title('RMSE across Folds')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('RMSE')
    ax2.grid(True, alpha=0.3)
    
    ax3.bar(folds, fold_results['mae_scores'])
    ax3.set_title('MAE across Folds')
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('MAE')
    ax3.grid(True, alpha=0.3)
    
    ax4.bar(folds, fold_results['r2_scores'])
    ax4.set_title('R² across Folds')
    ax4.set_xlabel('Fold')
    ax4.set_ylabel('R²')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'inverse_cv_metrics_by_fold_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Predicted vs Actual for all folds combined
    all_predictions = np.concatenate(fold_results['fold_predictions'])
    all_actuals = np.concatenate(fold_results['fold_actuals'])
    
    plt.figure(figsize=(10, 8))
    plt.scatter(all_actuals, all_predictions, alpha=0.6, s=20)
    
    # Perfect prediction line
    min_val = min(all_actuals.min(), all_predictions.min())
    max_val = max(all_actuals.max(), all_predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
    
    plt.xlabel('Actual Treated_astig (D)')
    plt.ylabel('Predicted Treated_astig (D)')
    plt.title('Inverse Prediction: Treated_astig (FIXED)\n(Combined across all CV folds)')
    plt.grid(True, alpha=0.3)
    
    # Add metrics text
    overall_mae = mean_absolute_error(all_actuals, all_predictions)
    overall_rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
    overall_r2 = r2_score(all_actuals, all_predictions)
    
    plt.text(0.05, 0.95, f'Overall MAE: {overall_mae:.4f}\nOverall RMSE: {overall_rmse:.4f}\nOverall R²: {overall_r2:.4f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig(output_dir / 'inverse_cv_predictions_vs_actual_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Residuals distribution
    residuals = all_predictions - all_actuals
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Residuals (Predicted - Actual Treated_astig)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Inverse Prediction Residuals (FIXED)')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    plt.savefig(output_dir / 'inverse_cv_residuals_distribution_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_cv_metrics_to_csv(fold_results, output_dir):
    """Save cross-validation metrics to CSV file in the same format as comparison tables."""
    
    # Extract metrics
    mse_scores = fold_results['mse_scores']
    rmse_scores = fold_results['rmse_scores']
    mae_scores = fold_results['mae_scores']
    r2_scores = fold_results['r2_scores']
    explained_var_scores = fold_results['explained_variance_scores']
    
    # Create DataFrame
    results_data = []
    
    # Add fold-wise results
    for i in range(len(mse_scores)):
        results_data.append({
            'Fold': i + 1,
            'MSE': mse_scores[i],
            'RMSE': rmse_scores[i],
            'MAE': mae_scores[i],
            'R²': r2_scores[i],
            'Explained Variance': explained_var_scores[i]
        })
    
    # Add mean values
    results_data.append({
        'Fold': 'Mean',
        'MSE': np.mean(mse_scores),
        'RMSE': np.mean(rmse_scores),
        'MAE': np.mean(mae_scores),
        'R²': np.mean(r2_scores),
        'Explained Variance': np.mean(explained_var_scores)
    })
    
    # Add standard deviation values
    results_data.append({
        'Fold': 'Std',
        'MSE': np.std(mse_scores),
        'RMSE': np.std(rmse_scores),
        'MAE': np.std(mae_scores),
        'R²': np.std(r2_scores),
        'Explained Variance': np.std(explained_var_scores)
    })
    
    # Add coefficient of variation (CV%) values
    results_data.append({
        'Fold': 'CV%',
        'MSE': (np.std(mse_scores) / np.mean(mse_scores)) * 100 if np.mean(mse_scores) != 0 else 0,
        'RMSE': (np.std(rmse_scores) / np.mean(rmse_scores)) * 100 if np.mean(rmse_scores) != 0 else 0,
        'MAE': (np.std(mae_scores) / np.mean(mae_scores)) * 100 if np.mean(mae_scores) != 0 else 0,
        'R²': (np.std(r2_scores) / np.mean(r2_scores)) * 100 if np.mean(r2_scores) != 0 else 0,
        'Explained Variance': (np.std(explained_var_scores) / np.mean(explained_var_scores)) * 100 if np.mean(explained_var_scores) != 0 else 0
    })
    
    # Create DataFrame and save to CSV
    df_results = pd.DataFrame(results_data)
    
    # Format numerical columns to 4 decimal places (except for CV% row)
    for col in ['MSE', 'RMSE', 'MAE', 'R²', 'Explained Variance']:
        df_results[col] = df_results[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
    
    # Save to CSV
    csv_filename = output_dir / 'Inverse_Treated_Astig_cv_fold_metrics_FIXED.csv'
    df_results.to_csv(csv_filename, index=False)
    
    return csv_filename

def main():
    """Main execution function."""
    print("="*70)
    print("INVERSE CROSS-VALIDATION ANALYSIS FOR TREATED_ASTIG PREDICTION - FIXED")
    print("="*70)
    
    # Load and prepare data
    print("\nLoading and preparing data...")
    df = load_and_prepare_data()
    
    # Define features and target
    features = ['Age', 'Steep_axis_term', 'WTW_IOLMaster', 'Treated_astig', 'Type', 'AL', 'LASIK?']
    target = 'Arcuate_sweep_total'
    
    X = df[features]
    y = df[target]
    
    print(f"Dataset shape: {X.shape}")
    print(f"Treated_astig range: {X['Treated_astig'].min():.2f} - {X['Treated_astig'].max():.2f}")
    
    # Define XGBoost parameters (same as in original model)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.02,
        'subsample': 0.85,
        'colsample_bytree': 0.9,
        'reg_alpha': 1.0,
        'reg_lambda': 5.0,
        'random_state': 42,
        'gamma': 0.1,
        'min_child_weight': 3,
        'monotone_constraints': (0, 0, 0, 1, 0, 0, 0)  # Monotonic constraint on Treated_astig
    }
    
    # Perform inverse CV analysis
    fold_results = perform_inverse_cv_analysis(X, y, params, n_splits=5)
    
    # Calculate and display summary statistics
    print("\n" + "="*50)
    print("INVERSE CROSS-VALIDATION RESULTS SUMMARY - FIXED")
    print("="*50)
    
    mse_scores = fold_results['mse_scores']
    rmse_scores = fold_results['rmse_scores']
    mae_scores = fold_results['mae_scores']
    r2_scores = fold_results['r2_scores']
    explained_var_scores = fold_results['explained_variance_scores']
    
    print(f"\nMSE across folds: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
    print(f"RMSE across folds: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    print(f"MAE across folds: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    print(f"R² across folds: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"Explained Variance across folds: {np.mean(explained_var_scores):.4f} ± {np.std(explained_var_scores):.4f}")
    
    print(f"\nIndividual fold results:")
    for i, (mse, rmse, mae, r2, ev) in enumerate(zip(mse_scores, rmse_scores, mae_scores, r2_scores, explained_var_scores), 1):
        print(f"  Fold {i}: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, EV={ev:.4f}")
    
    # Create visualizations
    print(f"\nGenerating plots...")
    plot_fold_results(fold_results, output_dir)
    
    # Save results to CSV
    print(f"\nSaving results to CSV...")
    csv_filename = save_cv_metrics_to_csv(fold_results, output_dir)
    
    print(f"\nFiles saved to {output_dir}:")
    print(f"  - inverse_cv_metrics_by_fold_fixed.png")
    print(f"  - inverse_cv_predictions_vs_actual_fixed.png")
    print(f"  - inverse_cv_residuals_distribution_fixed.png")
    print(f"  - {csv_filename.name}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE - FIXED VERSION")
    print("="*70)
    print("\nThis FIXED analysis uses:")
    print("• Realistic Treated_astig bounds from training data (fold-specific)")
    print("• Tolerance set to 0.01 for practical measurement precision")
    print("• Improved error handling and categorical feature management")
    print("\nThis should provide much more reasonable inverse prediction results!")

if __name__ == "__main__":
    main() 