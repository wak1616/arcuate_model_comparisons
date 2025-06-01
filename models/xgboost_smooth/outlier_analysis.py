"""
Outlier Analysis for XGBoost Smooth Models

This script identifies outliers (patients with large prediction errors) for both:
1. Forward analysis: predicting Arcuate_sweep_total from patient features
2. Backward analysis: predicting Treated_astig using inverse prediction

The script will output full patient information for outliers to help understand
what makes these cases difficult to predict accurately.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import shuffle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# Set up paths
current_dir = Path(__file__).parent
output_dir = current_dir
datasets_dir = Path(__file__).resolve().parents[2] / "data"

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

def train_forward_model(X, y):
    """Train the forward model (same as model4_xgboost_smooth.py)."""
    # Define XGBoost parameters
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
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)
    
    # Train model
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=300,
        early_stopping_rounds=20,
        evals=[(dtrain, 'train')],
        verbose_eval=False
    )
    
    return model, params

def predict_arcuate_for_astig(model, patient_base_features, treated_astig_value):
    """Predict Arcuate_sweep_total for a patient with a given Treated_astig value."""
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

def inverse_predict_treated_astig(model, patient_base_features, target_arcuate, astig_min, astig_max):
    """Find the Treated_astig value that produces the target Arcuate_sweep_total."""
    def objective(astig_val):
        try:
            predicted_arcuate = predict_arcuate_for_astig(model, patient_base_features, astig_val)
            return abs(predicted_arcuate - target_arcuate)
        except Exception:
            return 1000
    
    result = minimize_scalar(objective, bounds=(astig_min, astig_max), method='bounded')
    
    if result.success and result.fun < 1000:
        return result.x
    else:
        return (astig_min + astig_max) / 2

def identify_forward_outliers(df, model, percentile_threshold=95):
    """Identify outliers in forward prediction (Arcuate_sweep_total prediction)."""
    features = ['Age', 'Steep_axis_term', 'WTW_IOLMaster', 'Treated_astig', 'Type', 'AL', 'LASIK?']
    X = df[features]
    y = df['Arcuate_sweep_total']
    
    # Make predictions
    dmatrix = xgb.DMatrix(X, enable_categorical=True)
    predictions = model.predict(dmatrix)
    
    # Calculate residuals and absolute errors
    residuals = predictions - y.values
    abs_errors = np.abs(residuals)
    
    # Identify outliers based on percentile threshold
    error_threshold = np.percentile(abs_errors, percentile_threshold)
    outlier_indices = np.where(abs_errors >= error_threshold)[0]
    
    # Create outlier dataframe with additional information
    outlier_data = []
    for idx in outlier_indices:
        patient_data = df.iloc[idx].copy()
        patient_data['Predicted_Arcuate'] = predictions[idx]
        patient_data['Actual_Arcuate'] = y.iloc[idx]
        patient_data['Absolute_Error'] = abs_errors[idx]
        patient_data['Residual'] = residuals[idx]
        patient_data['Outlier_Index'] = idx
        outlier_data.append(patient_data)
    
    outliers_df = pd.DataFrame(outlier_data)
    
    return outliers_df, abs_errors, residuals, predictions

def identify_backward_outliers(df, model, percentile_threshold=95):
    """Identify outliers in backward prediction (Treated_astig prediction)."""
    features = ['Age', 'Steep_axis_term', 'WTW_IOLMaster', 'Treated_astig', 'Type', 'AL', 'LASIK?']
    X = df[features]
    y = df['Arcuate_sweep_total']
    
    # Get bounds for Treated_astig
    astig_min = max(X['Treated_astig'].min(), 0.05)
    astig_max = min(X['Treated_astig'].max(), 2.0)
    
    # Perform inverse prediction for each patient
    predicted_astig_values = []
    actual_astig_values = X['Treated_astig'].values
    
    print("Performing inverse predictions for backward analysis...")
    for i in range(len(df)):
        if i % 50 == 0:
            print(f"  Processing patient {i+1}/{len(df)}")
        
        # Get patient data
        patient_row = X.iloc[i]
        actual_arcuate = y.iloc[i]
        
        # Remove Treated_astig from patient features
        patient_base = patient_row.drop('Treated_astig')
        
        # Find Treated_astig value that would produce the actual Arcuate_sweep_total
        try:
            predicted_astig = inverse_predict_treated_astig(
                model, patient_base, actual_arcuate, astig_min, astig_max
            )
            predicted_astig_values.append(predicted_astig)
        except Exception:
            predicted_astig_values.append(np.median(X['Treated_astig']))
    
    predicted_astig_values = np.array(predicted_astig_values)
    
    # Calculate residuals and absolute errors
    residuals = predicted_astig_values - actual_astig_values
    abs_errors = np.abs(residuals)
    
    # Identify outliers based on percentile threshold
    error_threshold = np.percentile(abs_errors, percentile_threshold)
    outlier_indices = np.where(abs_errors >= error_threshold)[0]
    
    # Create outlier dataframe with additional information
    outlier_data = []
    for idx in outlier_indices:
        patient_data = df.iloc[idx].copy()
        patient_data['Predicted_Treated_astig'] = predicted_astig_values[idx]
        patient_data['Actual_Treated_astig'] = actual_astig_values[idx]
        patient_data['Absolute_Error'] = abs_errors[idx]
        patient_data['Residual'] = residuals[idx]
        patient_data['Outlier_Index'] = idx
        outlier_data.append(patient_data)
    
    outliers_df = pd.DataFrame(outlier_data)
    
    return outliers_df, abs_errors, residuals, predicted_astig_values

def save_outlier_summary(forward_outliers, backward_outliers, output_dir):
    """Save comprehensive outlier summaries to CSV files."""
    
    # Save forward outliers
    forward_outliers_sorted = forward_outliers.sort_values('Absolute_Error', ascending=False)
    forward_outliers_sorted.to_csv(output_dir / 'forward_analysis_outliers.csv', index=False)
    
    # Save backward outliers  
    backward_outliers_sorted = backward_outliers.sort_values('Absolute_Error', ascending=False)
    backward_outliers_sorted.to_csv(output_dir / 'backward_analysis_outliers.csv', index=False)
    
    # Create summary statistics
    summary_data = {
        'Analysis_Type': ['Forward (Arcuate Prediction)', 'Backward (Treated_astig Prediction)'],
        'Number_of_Outliers': [len(forward_outliers), len(backward_outliers)],
        'Mean_Absolute_Error': [forward_outliers['Absolute_Error'].mean(), backward_outliers['Absolute_Error'].mean()],
        'Max_Absolute_Error': [forward_outliers['Absolute_Error'].max(), backward_outliers['Absolute_Error'].max()],
        'Min_Absolute_Error': [forward_outliers['Absolute_Error'].min(), backward_outliers['Absolute_Error'].min()],
        'Std_Absolute_Error': [forward_outliers['Absolute_Error'].std(), backward_outliers['Absolute_Error'].std()]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'outlier_analysis_summary.csv', index=False)
    
    return forward_outliers_sorted, backward_outliers_sorted

def create_outlier_visualizations(forward_outliers, backward_outliers, 
                                forward_errors, backward_errors, output_dir):
    """Create visualizations for outlier analysis."""
    
    # Figure 1: Error distribution comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Forward analysis error distribution
    ax1.hist(forward_errors, bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(x=np.percentile(forward_errors, 95), color='red', linestyle='--', 
                label=f'95th percentile: {np.percentile(forward_errors, 95):.3f}')
    ax1.set_xlabel('Absolute Error in Arcuate Prediction')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Forward Analysis: Prediction Error Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Backward analysis error distribution
    ax2.hist(backward_errors, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(x=np.percentile(backward_errors, 95), color='red', linestyle='--',
                label=f'95th percentile: {np.percentile(backward_errors, 95):.3f}')
    ax2.set_xlabel('Absolute Error in Treated_astig Prediction')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Backward Analysis: Prediction Error Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'outlier_error_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Outlier characteristics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Forward outliers by Type
    if 'Type' in forward_outliers.columns:
        forward_outliers['Type'].value_counts().plot(kind='bar', ax=ax1)
        ax1.set_title('Forward Analysis Outliers by Type')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
    
    # Backward outliers by Type
    if 'Type' in backward_outliers.columns:
        backward_outliers['Type'].value_counts().plot(kind='bar', ax=ax2)
        ax2.set_title('Backward Analysis Outliers by Type')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
    
    # Forward outliers by LASIK status
    if 'LASIK?' in forward_outliers.columns:
        forward_outliers['LASIK?'].value_counts().plot(kind='bar', ax=ax3)
        ax3.set_title('Forward Analysis Outliers by LASIK Status')
        ax3.set_ylabel('Count')
        ax3.tick_params(axis='x', rotation=45)
    
    # Backward outliers by LASIK status
    if 'LASIK?' in backward_outliers.columns:
        backward_outliers['LASIK?'].value_counts().plot(kind='bar', ax=ax4)
        ax4.set_title('Backward Analysis Outliers by LASIK Status')
        ax4.set_ylabel('Count')
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'outlier_characteristics.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_outlier_analysis_report(forward_outliers, backward_outliers):
    """Print a comprehensive report of outlier analysis."""
    
    print("\n" + "="*80)
    print("OUTLIER ANALYSIS REPORT")
    print("="*80)
    
    print(f"\nFORWARD ANALYSIS (Arcuate Sweep Prediction):")
    print(f"  Total outliers identified: {len(forward_outliers)}")
    print(f"  Mean absolute error: {forward_outliers['Absolute_Error'].mean():.4f}")
    print(f"  Max absolute error: {forward_outliers['Absolute_Error'].max():.4f}")
    print(f"  Error range: {forward_outliers['Absolute_Error'].min():.4f} - {forward_outliers['Absolute_Error'].max():.4f}")
    
    print(f"\nBACKWARD ANALYSIS (Treated_astig Prediction):")
    print(f"  Total outliers identified: {len(backward_outliers)}")
    print(f"  Mean absolute error: {backward_outliers['Absolute_Error'].mean():.4f}")
    print(f"  Max absolute error: {backward_outliers['Absolute_Error'].max():.4f}")
    print(f"  Error range: {backward_outliers['Absolute_Error'].min():.4f} - {backward_outliers['Absolute_Error'].max():.4f}")
    
    # Show top 5 outliers for each analysis
    print(f"\nTOP 5 FORWARD ANALYSIS OUTLIERS:")
    print("-" * 50)
    forward_top5 = forward_outliers.nlargest(5, 'Absolute_Error')
    for idx, row in forward_top5.iterrows():
        print(f"Patient Index {int(row['Outlier_Index'])}:")
        print(f"  Age: {row['Age']}, Type: {row['Type']}, LASIK: {row['LASIK?']}")
        print(f"  Treated_astig: {row['Treated_astig']:.3f}, AL: {row['AL']:.2f}")
        print(f"  Actual Arcuate: {row['Actual_Arcuate']:.2f}, Predicted: {row['Predicted_Arcuate']:.2f}")
        print(f"  Absolute Error: {row['Absolute_Error']:.4f}")
        print()
    
    print(f"\nTOP 5 BACKWARD ANALYSIS OUTLIERS:")
    print("-" * 50)
    backward_top5 = backward_outliers.nlargest(5, 'Absolute_Error')
    for idx, row in backward_top5.iterrows():
        print(f"Patient Index {int(row['Outlier_Index'])}:")
        print(f"  Age: {row['Age']}, Type: {row['Type']}, LASIK: {row['LASIK?']}")
        print(f"  Actual Treated_astig: {row['Actual_Treated_astig']:.3f}, Predicted: {row['Predicted_Treated_astig']:.3f}")
        print(f"  Arcuate_sweep_total: {row['Arcuate_sweep_total']:.2f}, AL: {row['AL']:.2f}")
        print(f"  Absolute Error: {row['Absolute_Error']:.4f}")
        print()

def main():
    """Main execution function."""
    print("="*80)
    print("OUTLIER ANALYSIS FOR XGBOOST SMOOTH MODELS")
    print("="*80)
    
    # Load and prepare data
    print("\nLoading and preparing data...")
    df = load_and_prepare_data()
    
    # Define features
    features = ['Age', 'Steep_axis_term', 'WTW_IOLMaster', 'Treated_astig', 'Type', 'AL', 'LASIK?']
    target = 'Arcuate_sweep_total'
    
    X = df[features]
    y = df[target]
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {features}")
    
    # Train the forward model
    print("\nTraining forward prediction model...")
    model, params = train_forward_model(X, y)
    print("Forward model training completed.")
    
    # Identify forward outliers
    print("\nIdentifying forward analysis outliers...")
    forward_outliers, forward_errors, forward_residuals, forward_predictions = identify_forward_outliers(
        df, model, percentile_threshold=95
    )
    print(f"Found {len(forward_outliers)} forward analysis outliers (top 5% by error).")
    
    # Identify backward outliers
    print("\nIdentifying backward analysis outliers...")
    backward_outliers, backward_errors, backward_residuals, backward_predictions = identify_backward_outliers(
        df, model, percentile_threshold=95
    )
    print(f"Found {len(backward_outliers)} backward analysis outliers (top 5% by error).")
    
    # Save outlier data
    print("\nSaving outlier data to CSV files...")
    forward_sorted, backward_sorted = save_outlier_summary(forward_outliers, backward_outliers, output_dir)
    
    # Create visualizations
    print("\nCreating outlier visualizations...")
    create_outlier_visualizations(forward_outliers, backward_outliers, 
                                forward_errors, backward_errors, output_dir)
    
    # Print comprehensive report
    print_outlier_analysis_report(forward_outliers, backward_outliers)
    
    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nFiles saved to {output_dir}:")
    print(f"  - forward_analysis_outliers.csv (detailed forward outliers)")
    print(f"  - backward_analysis_outliers.csv (detailed backward outliers)")
    print(f"  - outlier_analysis_summary.csv (summary statistics)")
    print(f"  - outlier_error_distributions.png (error distribution plots)")
    print(f"  - outlier_characteristics.png (outlier characteristic plots)")
    print(f"\nThese files contain complete patient information for all identified outliers,")
    print(f"sorted by prediction error magnitude for easy review.")

if __name__ == "__main__":
    main() 