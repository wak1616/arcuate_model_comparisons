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
    
    # Perform cross-validation
    for train_idx, test_idx in cv.split(X):
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
        
        print(f"Fold {len(mse_scores)}: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    
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
        'EV_std': std_ev
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
    
    # Display results
    print("\nModel Comparison Results (with Cross-Validation):")
    print(formatted_df.to_string(index=False))

    # Save comparison results to file in the comparisons directory
    results_file = output_dir / 'model_cv_comparison_results.csv'
    formatted_df.to_csv(results_file, index=False)
    print(f"\nComparison results saved to '{results_file}'")

    # Create bar plots for model comparison
    if len(model_evaluations) > 1:
        # Create comparison plots for mean metrics
        metrics_to_plot = ['MSE', 'RMSE', 'MAE', 'R²']
        plt.figure(figsize=(15, 12))
        
        for i, metric in enumerate(metrics_to_plot):
            plt.subplot(2, 2, i+1)
            
            # Plot bars with error bars
            bars = plt.bar(
                comparison_df['Model'], 
                comparison_df[metric],
                yerr=comparison_df[f"{metric}_std"],
                capsize=10
            )
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:.4f}',
                    ha='center', va='bottom', 
                    rotation=0,
                    fontsize=9
                )
                
            plt.title(f'Comparison of {metric} across Models (with Std Dev)')
            plt.ylabel(metric)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=15)  # Rotate labels for better readability
            
        plt.tight_layout()
        # Save visualization to the comparisons directory
        viz_file = output_dir / 'model_cv_performance_comparison.png'
        plt.savefig(viz_file)
        print(f"Comparison visualizations saved to '{viz_file}'")
        
        # Create a special visualization to show prediction smoothness
        try:
            # Create a grid of Treated_astig values
            astig_range = np.linspace(X['Treated_astig'].min(), X['Treated_astig'].max(), 100)
            X_sample = X.sample(10, random_state=42).copy()
            
            # Function to predict across astig_range
            def predict_across_astig(model, X_base, astig_values):
                predictions = []
                for astig in astig_values:
                    X_temp = X_base.copy()
                    X_temp['Treated_astig'] = astig
                    
                    # Handle categorical features
                    X_temp['Type'] = X_temp['Type'].astype('category').cat.codes
                    X_temp['LASIK?'] = X_temp['LASIK?'].astype('category').cat.codes
                    
                    pred = model.predict(X_temp)
                    predictions.append(np.mean(pred))
                return predictions
            
            plt.figure(figsize=(12, 8))
            
            # Get predictions for each model across astig_range
            if 'xgb_model' in locals():
                standard_preds = predict_across_astig(xgb_model, X_sample, astig_range)
                plt.plot(astig_range, standard_preds, 'b-', label='XGBoost Standard', alpha=0.7)
            
            if 'xgb_monotonic_model' in locals():
                monotonic_preds = predict_across_astig(xgb_monotonic_model, X_sample, astig_range)
                plt.plot(astig_range, monotonic_preds, 'g-', label='XGBoost Monotonic', alpha=0.7)
            
            if 'xgb_smooth_model' in locals():
                smooth_preds = predict_across_astig(xgb_smooth_model, X_sample, astig_range)
                plt.plot(astig_range, smooth_preds, 'r-', linewidth=2.5, label='XGBoost Smooth-Monotonic')
            
            plt.title('Comparison of Model Smoothness\n(Predictions vs. Treated Astigmatism)')
            plt.xlabel('Treated Astigmatism (D)')
            plt.ylabel('Predicted Arcuate Sweep')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save the smoothness comparison
            smoothness_viz_file = output_dir / 'model_smoothness_comparison.png'
            plt.savefig(smoothness_viz_file)
            print(f"Smoothness comparison visualization saved to '{smoothness_viz_file}'")
            plt.close()
        except Exception as e:
            print(f"Error creating smoothness comparison: {e}")
    else:
        print("Only one model available for comparison, skipping visualization plots.")
else:
    print("No models available for comparison.") 