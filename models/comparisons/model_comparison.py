import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.model_selection import cross_val_score, KFold
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to sys.path to import from model directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Function to load and preprocess data
def load_data():
    # Get absolute path to the data file
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent  # Go up two levels to the project root
    data_file = project_root / "data" / "datafinal.csv"
    
    df = pd.read_csv(data_file, encoding='utf-8')
    
    # Basic preprocessing
    df['Type'] = df['Type'].str.strip()
    df['Sex'] = df['Sex'].str.strip()
    df['Eye'] = df['Eye'].str.strip()
    df['LASIK?'] = df['LASIK?'].str.strip()
    df['Type'] = df['Type'].replace('singe', 'single')
    

    # Set up features and target
    target = ['Arcuate_sweep_total']
    features = [
        'Age', 'Steep_axis_term', 'WTW_IOLMaster', 
        'Treated_astig', 'Type', 
        'AL', 'LASIK?'
    ]
    
    X = df[features]
    y = df[target]
    
    return X, y

# Function to perform cross-validation and evaluate a model
def evaluate_model_cv(model, X, y, model_name, n_folds=5, monotonic=False):
    print(f"\nPerforming {n_folds}-fold cross-validation for {model_name}...")
    
    # Define cross-validation strategy
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Initialize metrics lists
    mse_scores = []
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    ev_scores = []
    
    # Lists to store fold indices for plotting
    fold_indices = []
    
    # Perform cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X), 1):
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Apply preprocessing based on model type
        if model_name.startswith('XGBoost'):
            # Prepare data for XGBoost
            X_train_processed = X_train.copy()
            X_test_processed = X_test.copy()
            
            # Convert categorical features to codes
            X_train_processed['Type'] = X_train_processed['Type'].astype('category').cat.codes
            X_train_processed['LASIK?'] = X_train_processed['LASIK?'].astype('category').cat.codes
            X_test_processed['Type'] = X_test_processed['Type'].astype('category').cat.codes
            X_test_processed['LASIK?'] = X_test_processed['LASIK?'].astype('category').cat.codes
            
            # Train and predict
            model.fit(X_train_processed, y_train)
            y_pred = model.predict(X_test_processed)
            
        elif model_name == 'Random Forest':
            # Prepare data for Random Forest
            X_train_processed = X_train.copy()
            X_test_processed = X_test.copy()
            
            # One-hot encode categorical features
            categorical_features = ['Type', 'LASIK?']
            from sklearn.preprocessing import OneHotEncoder
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            
            # Apply one-hot encoding to categorical features
            cat_train = encoder.fit_transform(X_train_processed[categorical_features])
            cat_test = encoder.transform(X_test_processed[categorical_features])
            
            # Drop original categorical columns
            X_train_processed = X_train_processed.drop(categorical_features, axis=1)
            X_test_processed = X_test_processed.drop(categorical_features, axis=1)
            
            # Convert to numpy arrays and combine
            X_train_numeric = X_train_processed.values
            X_test_numeric = X_test_processed.values
            X_train_final = np.hstack((X_train_numeric, cat_train))
            X_test_final = np.hstack((X_test_numeric, cat_test))
            
            # Train and predict
            model.fit(X_train_final, y_train.values.ravel())
            y_pred = model.predict(X_test_final)
        elif model_name == 'Elastic Net':
            # For Elastic Net, the pipeline already handles preprocessing
            # Just pass the data directly to the model
            model.fit(X_train, y_train.values.ravel())
            y_pred = model.predict(X_test)
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        # Calculate metrics for this fold
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        ev = explained_variance_score(y_test, y_pred)
        
        mse_scores.append(mse)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        ev_scores.append(ev)
        fold_indices.append(fold_idx)
        
        print(f"Fold {fold_idx}: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    
    # Calculate average metrics
    avg_mse = np.mean(mse_scores)
    avg_rmse = np.mean(rmse_scores)
    avg_mae = np.mean(mae_scores)
    avg_r2 = np.mean(r2_scores)
    avg_ev = np.mean(ev_scores)
    
    # Calculate standard deviations
    std_mse = np.std(mse_scores)
    std_rmse = np.std(rmse_scores)
    std_mae = np.std(mae_scores)
    std_r2 = np.std(r2_scores)
    std_ev = np.std(ev_scores)
    
    print(f"\n{model_name} Cross-Validation Results (Mean ± Std):")
    print(f"MSE: {avg_mse:.4f} ± {std_mse:.4f}")
    print(f"RMSE: {avg_rmse:.4f} ± {std_rmse:.4f}")
    print(f"MAE: {avg_mae:.4f} ± {std_mae:.4f}")
    print(f"R²: {avg_r2:.4f} ± {std_r2:.4f}")
    print(f"Explained Variance: {avg_ev:.4f} ± {std_ev:.4f}")
    
    # Calculate coefficient of variation (CV) to measure relative stability
    # Lower values indicate more stable metrics across folds
    cv_mse = (std_mse / avg_mse) * 100 if avg_mse != 0 else float('inf')
    cv_rmse = (std_rmse / avg_rmse) * 100 if avg_rmse != 0 else float('inf')
    cv_mae = (std_mae / avg_mae) * 100 if avg_mae != 0 else float('inf')
    cv_r2 = (std_r2 / avg_r2) * 100 if avg_r2 != 0 else float('inf')
    
    print(f"\nMetric Stability (Coefficient of Variation %):")
    print(f"MSE CV: {cv_mse:.2f}% (lower is more stable)")
    print(f"RMSE CV: {cv_rmse:.2f}% (lower is more stable)")
    print(f"MAE CV: {cv_mae:.2f}% (lower is more stable)")
    print(f"R² CV: {cv_r2:.2f}% (lower is more stable)")
    
    return {
        'Model': model_name,
        'MSE': avg_mse,
        'MSE_std': std_mse,
        'RMSE': avg_rmse,
        'RMSE_std': std_rmse,
        'MAE': avg_mae,
        'MAE_std': std_mae,
        'R²': avg_r2,
        'R²_std': std_r2,
        'Explained Variance': avg_ev,
        'EV_std': std_ev,
        'Fold_Indices': fold_indices,
        'MSE_per_fold': mse_scores,
        'RMSE_per_fold': rmse_scores,
        'MAE_per_fold': mae_scores,
        'R2_per_fold': r2_scores,
        'EV_per_fold': ev_scores,
        'MSE_CV': cv_mse,
        'RMSE_CV': cv_rmse,
        'MAE_CV': cv_mae,
        'R2_CV': cv_r2
    }

# Set up output directory for saving files
current_dir = Path(__file__).parent
output_dir = current_dir
os.makedirs(output_dir, exist_ok=True)

# Load the data
X, y = load_data()

# Dictionary to store results
model_evaluations = []

# Try evaluating XGBoost model (from the xgboost directory)
try:
    print("Loading XGBoost model...")
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../xgboost')))
    import xgboost as xgb
    from xgboost import XGBRegressor
    
    # Create XGBoost model
    xgb_model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    
    # Perform cross-validation and evaluate
    xgb_eval = evaluate_model_cv(xgb_model, X, y, 'XGBoost')
    model_evaluations.append(xgb_eval)
    print("XGBoost model evaluated successfully.")
except Exception as e:
    print(f"Error evaluating XGBoost model: {e}")

# Try evaluating XGBoost monotonic model (from the xgboost_monotonic directory)
try:
    print("Loading XGBoost Monotonic model...")
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../xgboost_monotonic')))
    import xgboost as xgb
    from xgboost import XGBRegressor
    
    # Create XGBoost monotonic model with selected monotonicity constraint
    xgb_monotonic_model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=1.0,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=42,
        gamma=0.1,
        monotone_constraints=(0, 0, 0, 1, 0, 0, 0)  # Only Treated_astig has monotonic constraint
        # Age(0), Steep_axis_term(0), WTW_IOLMaster(0), Treated_astig(+), Type(0), AL(0), LASIK?(0)
    )
    
    # Perform cross-validation and evaluate
    xgb_monotonic_eval = evaluate_model_cv(xgb_monotonic_model, X, y, 'XGBoost Selective-Monotonic')
    model_evaluations.append(xgb_monotonic_eval)
    print("XGBoost Selective-Monotonic model evaluated successfully.")
except Exception as e:
    print(f"Error evaluating XGBoost Selective-Monotonic model: {e}")

# Try evaluating Smooth XGBoost model (model4 from the xgboost_smooth directory)
try:
    print("Loading XGBoost Smooth model...")
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../xgboost_smooth')))
    import xgboost as xgb
    from xgboost import XGBRegressor
    
    # Create XGBoost smooth model with fine-tuned parameters for smoother predictions
    xgb_smooth_model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.02,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=42,
        gamma=0.1,
        min_child_weight=3,
        monotone_constraints=(0, 0, 0, 1, 0, 0, 0)  # Only Treated_astig has monotonic constraint
    )
    
    # Perform cross-validation and evaluate
    xgb_smooth_eval = evaluate_model_cv(xgb_smooth_model, X, y, 'XGBoost Smooth-Monotonic')
    model_evaluations.append(xgb_smooth_eval)
    print("XGBoost Smooth-Monotonic model evaluated successfully.")
except Exception as e:
    print(f"Error evaluating XGBoost Smooth-Monotonic model: {e}")

# Try evaluating Random Forest model (from the random_forest directory)
try:
    print("Loading Random Forest model...")
    from sklearn.ensemble import RandomForestRegressor
    
    # Create Random Forest model
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=2,
        random_state=42
    )
    
    # Perform cross-validation and evaluate
    rf_eval = evaluate_model_cv(rf_model, X, y, 'Random Forest')
    model_evaluations.append(rf_eval)
    print("Random Forest model evaluated successfully.")
except Exception as e:
    print(f"Error evaluating Random Forest model: {e}")

# Try evaluating Elastic Net model (from the elastic_net directory)
try:
    print("Loading Elastic Net model...")
    elastic_net_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../elastic_net/elastic_net_best_model.pkl'))
    
    if os.path.exists(elastic_net_model_path):
        # Load the pre-trained model
        elastic_net_model = joblib.load(elastic_net_model_path)
        
        # Perform cross-validation and evaluate
        elastic_net_eval = evaluate_model_cv(elastic_net_model, X, y, 'Elastic Net')
        model_evaluations.append(elastic_net_eval)
        print("Elastic Net model evaluated successfully.")
    else:
        print(f"Elastic Net model file not found at {elastic_net_model_path}")
except Exception as e:
    print(f"Error evaluating Elastic Net model: {e}")

# Try evaluating Neural Network model (model5 from the neural_net directory)
try:
    print("Loading Neural Network model...")
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../neural_net')))
    import torch
    import torch.nn as nn
    
    # Define the MonotonicLayer class needed to load the model
    class MonotonicLayer(nn.Module):
        """Custom layer that ensures monotonicity by using positive weights"""
        def __init__(self, in_features, out_features):
            super(MonotonicLayer, self).__init__()
            # Initialize weights but ensure they stay positive
            self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
            self.bias = nn.Parameter(torch.zeros(out_features))
            
        def forward(self, x):
            # Apply ReLU to weights to force them to be positive
            positive_weights = torch.relu(self.weight)
            return torch.matmul(x, positive_weights.t()) + self.bias
    
    # Define the HybridMonotonicMLP class needed to load the model
    class HybridMonotonicMLP(nn.Module):
        """
        A hybrid neural network with two branches:
        1. Monotonic branch: Processes Treated_astig with monotonicity constraint
        2. Standard MLP branch: Processes all other features
        The branches are combined for the final prediction
        """
        def __init__(self, num_numeric, num_categorical_encoded, monotonic_feature_idx,
                    hidden_size=32, dropout_rate=0.2):
            super(HybridMonotonicMLP, self).__init__()
            
            # Architecture dimensions
            self.num_numeric = num_numeric
            self.num_categorical_encoded = num_categorical_encoded
            self.monotonic_feature_idx = monotonic_feature_idx
            self.hidden_size = hidden_size
            
            # Monotonic Path (for Treated_astig)
            self.monotonic_path = nn.Sequential(
                MonotonicLayer(1, hidden_size),
                nn.ReLU(),
                MonotonicLayer(hidden_size, hidden_size // 2),
                nn.ReLU(),
                MonotonicLayer(hidden_size // 2, 1)
            )
            
            # Standard MLP Path (for all other features)
            total_other_features = num_numeric - 1 + num_categorical_encoded
            self.standard_mlp = nn.Sequential(
                nn.Linear(total_other_features, hidden_size * 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1)
            )
            
            # Final combination layer
            self.combiner = nn.Linear(2, 1)
            
        def forward(self, x):
            # Split features
            monotonic_feature = x[:, self.monotonic_feature_idx].unsqueeze(1)  # Treated_astig
            
            # Create mask for other features (all numeric except monotonic + all categorical)
            other_features_idx = list(range(self.num_numeric))
            other_features_idx.remove(self.monotonic_feature_idx)
            other_features_idx.extend(range(self.num_numeric, self.num_numeric + self.num_categorical_encoded))
            
            other_features = x[:, other_features_idx]
            
            # Process through respective paths
            monotonic_output = self.monotonic_path(monotonic_feature)
            standard_output = self.standard_mlp(other_features)
            
            # Combine the outputs
            combined = torch.cat((monotonic_output, standard_output), dim=1)
            output = self.combiner(combined)
            
            return output
    
    # Load the Neural Network model
    nn_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../neural_net/neural_network_model.pt'))
    
    if os.path.exists(nn_model_path):
        # Add safe globals for sklearn components
        torch.serialization.add_safe_globals(['sklearn.compose._column_transformer.ColumnTransformer'])
        
        # Define a wrapper class for compatibility with sklearn-style models
        class NeuralNetworkWrapper:
            def __init__(self, model_path):
                # Use weights_only=False to load the full checkpoint including preprocessor
                self.checkpoint = torch.load(model_path, weights_only=False)
                
                # Extract model configuration
                config = self.checkpoint['model_config']
                self.model = HybridMonotonicMLP(
                    num_numeric=config['num_numeric'],
                    num_categorical_encoded=config['num_categorical_encoded'],
                    monotonic_feature_idx=config['monotonic_feature_idx'],
                    hidden_size=config['hidden_size'],
                    dropout_rate=config['dropout_rate']
                )
                
                # Load model weights
                self.model.load_state_dict(self.checkpoint['model_state_dict'])
                self.model.eval()
                
                # Load preprocessor
                self.preprocessor = self.checkpoint['preprocessor']
                
            def fit(self, X, y):
                # This is just a placeholder, as the model is already trained
                return self
            
            def predict(self, X):
                # Preprocess the input
                X_processed = self.preprocessor.transform(X)
                X_tensor = torch.FloatTensor(X_processed)
                
                # Make predictions
                with torch.no_grad():
                    pred = self.model(X_tensor).numpy()
                
                return pred
                
        # Create the wrapper model
        nn_wrapper = NeuralNetworkWrapper(nn_model_path)
        
        # Perform cross-validation and evaluate
        nn_eval = evaluate_model_cv(nn_wrapper, X, y, 'Neural Network Monotonic')
        model_evaluations.append(nn_eval)
        print("Neural Network model evaluated successfully.")
    else:
        print(f"Neural Network model file not found at {nn_model_path}")
except Exception as e:
    print(f"Error evaluating Neural Network model: {e}")

# Create a comparison DataFrame
comparison_df = pd.DataFrame(model_evaluations)

if len(model_evaluations) > 0:
    # Format the comparison results with mean ± std
    formatted_df = pd.DataFrame()
    formatted_df['Model'] = comparison_df['Model']
    
    # Format metrics with standard deviations
    for metric in ['MSE', 'RMSE', 'MAE', 'R²']:
        formatted_df[metric] = comparison_df.apply(
            lambda row: f"{row[metric]:.4f} ± {row[metric+'_std']:.4f}", 
            axis=1
        )
    
    # Handle Explained Variance separately since the key name might be different
    formatted_df['Explained Variance'] = comparison_df.apply(
        lambda row: f"{row['Explained Variance']:.4f} ± {row['EV_std']:.4f}", 
        axis=1
    )
    
    # Add coefficient of variation (stability metric)
    for metric in ['MSE_CV', 'RMSE_CV', 'MAE_CV', 'R2_CV']:
        formatted_df[metric.replace('_CV', ' CV%')] = comparison_df[metric].round(2).astype(str) + '%'
    
    # Display results
    print("\nModel Comparison Results (with Cross-Validation):")
    print(formatted_df.to_string(index=False))

    # Save comparison results to file in the comparisons directory
    results_file = output_dir / 'model_cv_comparison_results.csv'
    formatted_df.to_csv(results_file, index=False)
    print(f"\nComparison results saved to '{results_file}'")

    # Create detailed CV fold tables for each model
    cv_tables_dir = output_dir / 'cv_fold_tables'
    os.makedirs(cv_tables_dir, exist_ok=True)
    
    for model_eval in model_evaluations:
        model_name = model_eval['Model']
        fold_metrics = pd.DataFrame({
            'Fold': model_eval['Fold_Indices'],
            'MSE': model_eval['MSE_per_fold'],
            'RMSE': model_eval['RMSE_per_fold'],
            'MAE': model_eval['MAE_per_fold'],
            'R²': model_eval['R2_per_fold'],
            'Explained Variance': model_eval['EV_per_fold']
        })
        
        # Round metrics for display
        fold_metrics = fold_metrics.round(4)
        
        # Add a row with mean and std
        mean_row = pd.DataFrame({
            'Fold': ['Mean'],
            'MSE': [model_eval['MSE']],
            'RMSE': [model_eval['RMSE']],
            'MAE': [model_eval['MAE']],
            'R²': [model_eval['R²']],
            'Explained Variance': [model_eval['Explained Variance']]
        }).round(4)
        
        std_row = pd.DataFrame({
            'Fold': ['Std'],
            'MSE': [model_eval['MSE_std']],
            'RMSE': [model_eval['RMSE_std']],
            'MAE': [model_eval['MAE_std']],
            'R²': [model_eval['R²_std']],
            'Explained Variance': [model_eval['EV_std']]
        }).round(4)
        
        cv_row = pd.DataFrame({
            'Fold': ['CV%'],
            'MSE': [model_eval['MSE_CV']],
            'RMSE': [model_eval['RMSE_CV']],
            'MAE': [model_eval['MAE_CV']],
            'R²': [model_eval['R2_CV']],
            'Explained Variance': [np.nan]  # No CV for EV
        }).round(2)
        
        fold_metrics = pd.concat([fold_metrics, mean_row, std_row, cv_row], ignore_index=True)
        
        # Save to CSV
        fold_metrics.to_csv(cv_tables_dir / f"{model_name.replace(' ', '_')}_cv_fold_metrics.csv", index=False)
        
        # Create fold metrics plot for this model
        plt.figure(figsize=(12, 10))
        
        metrics_to_plot = ['MSE', 'RMSE', 'MAE', 'R²']
        for i, metric in enumerate(metrics_to_plot):
            plt.subplot(2, 2, i+1)
            
            # Get values excluding the Mean, Std, and CV% rows
            fold_values = model_eval[f"{metric if metric != 'R²' else 'R2'}_per_fold"]
            mean_value = model_eval[metric if metric != 'R²' else 'R²']
            std_value = model_eval[f"{metric if metric != 'R²' else 'R²'}_std"]
            
            # Plot individual fold values
            plt.bar(model_eval['Fold_Indices'], fold_values, alpha=0.7)
            
            # Plot mean as a horizontal line
            plt.axhline(y=mean_value, color='r', linestyle='-', label=f'Mean: {mean_value:.4f}')
            
            # Plot standard deviation range
            plt.axhline(y=mean_value + std_value, color='g', linestyle='--', 
                       label=f'Mean ± Std: {std_value:.4f}')
            plt.axhline(y=mean_value - std_value, color='g', linestyle='--')
            
            plt.title(f'{metric} Across {len(fold_values)} Folds for {model_name}')
            plt.xlabel('Fold Number')
            plt.ylabel(metric)
            plt.xticks(model_eval['Fold_Indices'])
            plt.grid(True, alpha=0.3)
            plt.legend()
            
        plt.tight_layout()
        plt.savefig(cv_tables_dir / f"{model_name.replace(' ', '_')}_cv_fold_metrics.png")
        plt.close()
    
    # Create a comparison visualization of CV stability across models
    if len(model_evaluations) > 1:
        plt.figure(figsize=(14, 10))
        
        # Extract coefficient of variation data for comparison
        models = [eval_dict['Model'] for eval_dict in model_evaluations]
        mse_cv = [eval_dict['MSE_CV'] for eval_dict in model_evaluations]
        rmse_cv = [eval_dict['RMSE_CV'] for eval_dict in model_evaluations]
        mae_cv = [eval_dict['MAE_CV'] for eval_dict in model_evaluations]
        r2_cv = [eval_dict['R2_CV'] for eval_dict in model_evaluations]
        
        # Create a DataFrame for easier plotting
        cv_df = pd.DataFrame({
            'Model': models,
            'MSE CV%': mse_cv,
            'RMSE CV%': rmse_cv,
            'MAE CV%': mae_cv,
            'R² CV%': r2_cv
        })
        
        # Plot coefficient of variation for each metric across models
        metrics_cv = ['MSE CV%', 'RMSE CV%', 'MAE CV%', 'R² CV%']
        for i, metric in enumerate(metrics_cv):
            plt.subplot(2, 2, i+1)
            bars = plt.bar(cv_df['Model'], cv_df[metric], alpha=0.7)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:.2f}%',
                    ha='center', va='bottom', 
                    rotation=0,
                    fontsize=9
                )
            
            plt.title(f'{metric} Across Models (Lower is More Stable)')
            plt.ylabel('Coefficient of Variation (%)')
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            plt.xticks(rotation=15)
            
        plt.tight_layout()
        plt.savefig(output_dir / 'model_cv_stability_comparison.png')
        plt.close()
        
        # Create detailed fold-by-fold comparison across models
        plt.figure(figsize=(15, 12))
        
        # Plot each metric across all folds for all models
        metrics_to_compare = ['RMSE', 'MAE', 'R²']
        for i, metric in enumerate(metrics_to_compare):
            plt.subplot(len(metrics_to_compare), 1, i+1)
            
            metric_key = 'R2' if metric == 'R²' else metric
            
            for model_eval in model_evaluations:
                plt.plot(
                    model_eval['Fold_Indices'], 
                    model_eval[f'{metric_key}_per_fold'],
                    'o-', 
                    label=model_eval['Model'],
                    alpha=0.7
                )
            
            plt.title(f'{metric} Across All Folds for All Models')
            plt.xlabel('Fold Number')
            plt.ylabel(metric)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
            
        plt.tight_layout()
        plt.savefig(output_dir / 'model_cv_fold_comparison.png')
        plt.close()
        
        # Create a heatmap of fold metrics for each model
        for metric in ['RMSE', 'MAE', 'R²']:
            metric_key = 'R2' if metric == 'R²' else metric
            
            # Prepare data for heatmap
            heatmap_data = []
            model_names = []
            
            for model_eval in model_evaluations:
                model_names.append(model_eval['Model'])
                heatmap_data.append(model_eval[f'{metric_key}_per_fold'])
            
            # Create heatmap
            plt.figure(figsize=(12, len(model_names) * 0.8 + 2))
            
            # Convert to numpy array and transpose to get models as rows and folds as columns
            heatmap_array = np.array(heatmap_data)
            
            # Create a custom colormap
            if metric == 'R²':
                # For R², higher is better (use viridis)
                cmap = 'viridis'
                # Set vmin and vmax to ensure 0 is a distinct color
                vmin = max(0, heatmap_array.min() - 0.1)
                vmax = min(1, heatmap_array.max() + 0.1)
            else:
                # For error metrics, lower is better (use viridis_r)
                cmap = 'viridis_r'
                # Set vmin and vmax to ensure proper color scale
                vmin = max(0, heatmap_array.min() - heatmap_array.std())
                vmax = heatmap_array.max() + heatmap_array.std()
            
            # Create heatmap
            ax = sns.heatmap(
                heatmap_array,
                annot=True, 
                fmt=".4f",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                yticklabels=model_names,
                xticklabels=[f'Fold {i}' for i in range(1, 6)],
                cbar_kws={'label': metric}
            )
            
            plt.title(f'{metric} Comparison Across Models and Folds')
            plt.tight_layout()
            plt.savefig(output_dir / f'model_cv_{metric_key}_heatmap.png')
            plt.close()

        print(f"\nDetailed cross-validation fold tables and visualizations saved in '{cv_tables_dir}'")
        print(f"Stability comparison saved as '{output_dir / 'model_cv_stability_comparison.png'}'")
        print(f"Fold-by-fold comparison saved as '{output_dir / 'model_cv_fold_comparison.png'}'")
        print(f"Metric heatmaps saved in the output directory")
    else:
        print("Only one model available for comparison, skipping cross-model stability visualizations.")
else:
    print("No models available for comparison.") 