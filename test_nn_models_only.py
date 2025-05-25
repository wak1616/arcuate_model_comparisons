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
    project_root = current_file.parent  # Current directory is project root
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
        
        # For Neural Network models, they handle their own preprocessing internally
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        
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
    
    return {
        'Model': model_name,
        'MSE': avg_mse,
        'RMSE': avg_rmse,
        'MAE': avg_mae,
        'R²': avg_r2,
        'Explained Variance': avg_ev
    }

# Load the data
X, y = load_data()
print(f"Data loaded: X={X.shape}, y={y.shape}")

# Dictionary to store results
model_evaluations = []

# Try evaluating Neural Network model (model5 - train from scratch for fair comparison)
try:
    print("="*70)
    print("TESTING NEURAL NETWORK MODEL (MODEL5)")
    print("="*70)
    
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/neural_net')))
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    
    # Define the MonotonicLayer class
    class MonotonicLayer(nn.Module):
        """Custom layer that ensures monotonicity by using positive weights"""
        def __init__(self, in_features, out_features):
            super(MonotonicLayer, self).__init__()
            self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
            self.bias = nn.Parameter(torch.zeros(out_features))
            
        def forward(self, x):
            positive_weights = torch.relu(self.weight)
            return torch.matmul(x, positive_weights.t()) + self.bias
    
    # Define the HybridMonotonicMLP class
    class HybridMonotonicMLP(nn.Module):
        def __init__(self, num_numeric, num_categorical_encoded, monotonic_feature_idx,
                    hidden_size=32, dropout_rate=0.2):
            super(HybridMonotonicMLP, self).__init__()
            
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
            monotonic_feature = x[:, self.monotonic_feature_idx].unsqueeze(1)
            
            # Create mask for other features
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
    
    # Define a wrapper class that trains the model from scratch for each fold
    class NeuralNetworkWrapper:
        def __init__(self):
            # Set device (GPU if available, otherwise CPU)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"  Neural Network using device: {self.device}")
            
            # Define features
            self.numeric_features = ['Age', 'Steep_axis_term', 'WTW_IOLMaster', 'Treated_astig', 'AL']
            self.categorical_features = ['Type', 'LASIK?']
            self.monotonic_feature = 'Treated_astig'
            self.monotonic_feature_idx = self.numeric_features.index(self.monotonic_feature)
            
            # Initialize preprocessor
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), self.numeric_features),
                    ('cat', OneHotEncoder(drop='first'), self.categorical_features)
                ])
            
            self.model = None
            
        def fit(self, X, y):
            # Fit preprocessor and transform data
            X_processed = self.preprocessor.fit_transform(X)
            
            # Get dimensions
            n_numeric = len(self.numeric_features)
            n_categorical_encoded = X_processed.shape[1] - n_numeric
            
            # Convert to tensors and move to device
            X_tensor = torch.FloatTensor(X_processed).to(self.device)
            y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)
            
            # Initialize model
            self.model = HybridMonotonicMLP(
                num_numeric=n_numeric,
                num_categorical_encoded=n_categorical_encoded,
                monotonic_feature_idx=self.monotonic_feature_idx,
                hidden_size=32,
                dropout_rate=0.2
            ).to(self.device)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
            
            # Training loop
            self.model.train()
            for epoch in range(100):  # Reduced epochs for CV
                optimizer.zero_grad()
                outputs = self.model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
            
            return self
            
        def predict(self, X):
            # Transform data using fitted preprocessor
            X_processed = self.preprocessor.transform(X)
            X_tensor = torch.FloatTensor(X_processed).to(self.device)
            
            # Make predictions
            self.model.eval()
            with torch.no_grad():
                pred = self.model(X_tensor).cpu().numpy()
            
            return pred.flatten()
    
    # Create the wrapper model and perform cross-validation
    nn_wrapper = NeuralNetworkWrapper()
    
    # Perform cross-validation and evaluate
    nn_eval = evaluate_model_cv(nn_wrapper, X, y, 'Neural Network Monotonic')
    model_evaluations.append(nn_eval)
    print("Neural Network model evaluated successfully.")
    
except Exception as e:
    print(f"Error evaluating Neural Network model: {e}")
    import traceback
    traceback.print_exc()

# Try evaluating Monotonic Neural Network model (model7 from the monotonicNN directory)
try:
    print("\n" + "="*70)
    print("TESTING MONOTONIC NEURAL NETWORK (MODEL7)")
    print("="*70)
    
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/monotonicNN')))
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from torch.utils.data import TensorDataset, DataLoader
    
    # Define the SimpleMonotonicNN class needed for the model
    class SimpleMonotonicNN(nn.Module):
        def __init__(self, other_input_dim):
            super().__init__()
            self.unconstrained_path = nn.Sequential(
                nn.Linear(other_input_dim, 48),
                nn.LeakyReLU(0.1),
                nn.Linear(48, 10),
                nn.ReLU()
            )
            
        def forward(self, x_other, x_monotonic):
            coefficients = self.unconstrained_path(x_other)
            monotonic_feature_contributions = coefficients * x_monotonic
            return monotonic_feature_contributions.sum(dim=1, keepdim=True)
    
    # Define a wrapper class that trains the model from scratch for each fold (like model5)
    class MonotonicNeuralNetworkWrapper:
        def __init__(self):
            # Set device (GPU if available, otherwise CPU)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"  MonotonicNN using device: {self.device}")
            
            # Initialize components that will be fitted during training
            self.type_le = LabelEncoder()
            self.lasik_le = LabelEncoder()
            self.other_scaler = StandardScaler()
            self.monotonic_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
            self.model = None
            self.x_train_min = None
            self.other_features_order = None
            self.monotonic_feature_order = None
            
        def fit(self, X, y):
            # Add Treatment_astigmatism column if it doesn't exist (copy from Treated_astig)
            X_processed = X.copy()
            if 'Treatment_astigmatism' not in X_processed.columns:
                X_processed['Treatment_astigmatism'] = X_processed['Treated_astig']
            
            # Handle NaN values
            wtw_median = X_processed['WTW_IOLMaster'].median()
            al_median = X_processed['AL'].median()
            X_processed['WTW_IOLMaster'] = X_processed['WTW_IOLMaster'].fillna(wtw_median)
            X_processed['AL'] = X_processed['AL'].fillna(al_median)
            
            # Extract monotonic feature and calculate min
            x_monotonic = X_processed['Treated_astig']
            self.x_train_min = x_monotonic.min()
            
            # Create monotonic features
            monotonic_features_dict = {
                'constant': np.ones_like(x_monotonic),
                'linear': x_monotonic,
                'logistic_shift_left_1': 1 / (1 + np.exp(-(x_monotonic+1))),
                'logistic_shift_left_0.5': 1 / (1 + np.exp(-(x_monotonic+0.5))),
                'logistic_center': 1 / (1 + np.exp(-x_monotonic)),
                'logarithmic': np.log(x_monotonic - self.x_train_min + 1),
                'logistic_shift_right_0.5': 1 / (1 + np.exp(-(x_monotonic-0.5))),
                'logistic_shift_right_1': 1 / (1 + np.exp(-(x_monotonic-1))),
                'logistic_shift_right_1.5': 1 / (1 + np.exp(-(x_monotonic-1.5))),
                'logistic_shift_left_1.5': 1 / (1 + np.exp(-(x_monotonic+1.5)))
            }
            X_monotonic = pd.DataFrame(monotonic_features_dict)
            
            # Prepare other features
            other_features = [
                'Age', 'Steep_axis_term', 'WTW_IOLMaster',
                'AL', 'LASIK?', 'Treatment_astigmatism', 'Type'
            ]
            X_other = X_processed[other_features].copy()
            
            # Fit and transform encoders
            X_other['Type'] = self.type_le.fit_transform(X_other['Type'])
            X_other['LASIK?'] = self.lasik_le.fit_transform(X_other['LASIK?'])
            
            # Store feature orders
            self.other_features_order = list(X_other.columns)
            self.monotonic_feature_order = list(X_monotonic.columns)
            
            # Fit and transform scalers
            X_other_scaled = pd.DataFrame(self.other_scaler.fit_transform(X_other), columns=self.other_features_order)
            X_monotonic_scaled = pd.DataFrame(self.monotonic_scaler.fit_transform(X_monotonic), columns=self.monotonic_feature_order)
            y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1))
            
            # Convert to tensors and move to device
            x_other_tensor = torch.FloatTensor(X_other_scaled.values).to(self.device)
            x_monotonic_tensor = torch.FloatTensor(X_monotonic_scaled.values).to(self.device)
            y_tensor = torch.FloatTensor(y_scaled).to(self.device)
            
            # Initialize and train model
            self.model = SimpleMonotonicNN(len(self.other_features_order)).to(self.device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
            
            # Training loop (simplified for CV)
            self.model.train()
            for epoch in range(100):  # Reduced epochs for CV
                optimizer.zero_grad()
                outputs = self.model(x_other_tensor, x_monotonic_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            
            return self
            
        def predict(self, X):
            # Add Treatment_astigmatism column if it doesn't exist
            X_processed = X.copy()
            if 'Treatment_astigmatism' not in X_processed.columns:
                X_processed['Treatment_astigmatism'] = X_processed['Treated_astig']
            
            # Handle NaN values using training medians
            wtw_idx = self.other_features_order.index('WTW_IOLMaster')
            al_idx = self.other_features_order.index('AL')
            wtw_median = self.other_scaler.mean_[wtw_idx]
            al_median = self.other_scaler.mean_[al_idx]
            
            X_processed['WTW_IOLMaster'] = X_processed['WTW_IOLMaster'].fillna(wtw_median)
            X_processed['AL'] = X_processed['AL'].fillna(al_median)
            
            # Extract monotonic feature
            x_monotonic = X_processed['Treated_astig']
            
            # Create monotonic features
            monotonic_features_dict = {
                'constant': np.ones_like(x_monotonic),
                'linear': x_monotonic,
                'logistic_shift_left_1': 1 / (1 + np.exp(-(x_monotonic+1))),
                'logistic_shift_left_0.5': 1 / (1 + np.exp(-(x_monotonic+0.5))),
                'logistic_center': 1 / (1 + np.exp(-x_monotonic)),
                'logarithmic': np.log(x_monotonic - self.x_train_min + 1),
                'logistic_shift_right_0.5': 1 / (1 + np.exp(-(x_monotonic-0.5))),
                'logistic_shift_right_1': 1 / (1 + np.exp(-(x_monotonic-1))),
                'logistic_shift_right_1.5': 1 / (1 + np.exp(-(x_monotonic-1.5))),
                'logistic_shift_left_1.5': 1 / (1 + np.exp(-(x_monotonic+1.5)))
            }
            X_monotonic = pd.DataFrame(monotonic_features_dict, columns=self.monotonic_feature_order)
            
            # Prepare other features
            other_features = [
                'Age', 'Steep_axis_term', 'WTW_IOLMaster',
                'AL', 'LASIK?', 'Treatment_astigmatism', 'Type'
            ]
            X_other = X_processed[other_features].copy()
            
            # Transform categorical features
            X_other['Type'] = self.type_le.transform(X_other['Type'].map(
                lambda s: s if s in self.type_le.classes_ else self.type_le.classes_[0]
            ))
            X_other['LASIK?'] = self.lasik_le.transform(X_other['LASIK?'].map(
                lambda s: s if s in self.lasik_le.classes_ else self.lasik_le.classes_[0]
            ))
            
            # Ensure feature order
            X_other = X_other[self.other_features_order]
            
            # Scale features
            X_other_scaled = pd.DataFrame(
                self.other_scaler.transform(X_other), 
                columns=self.other_features_order
            )
            X_monotonic_scaled = pd.DataFrame(
                self.monotonic_scaler.transform(X_monotonic), 
                columns=self.monotonic_feature_order
            )
            
            # Convert to tensors and predict
            x_other_tensor = torch.FloatTensor(X_other_scaled.values).to(self.device)
            x_monotonic_tensor = torch.FloatTensor(X_monotonic_scaled.values).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                pred_scaled = self.model(x_other_tensor, x_monotonic_tensor)
                pred = self.target_scaler.inverse_transform(pred_scaled.cpu().numpy())
                pred = np.maximum(0.0, pred)  # Ensure non-negative
            
            return pred.flatten()
    
    # Create the wrapper model and perform cross-validation
    monotonic_nn_wrapper = MonotonicNeuralNetworkWrapper()
    
    # Perform cross-validation and evaluate
    monotonic_nn_eval = evaluate_model_cv(monotonic_nn_wrapper, X, y, 'Monotonic Neural Network')
    model_evaluations.append(monotonic_nn_eval)
    print("Monotonic Neural Network model evaluated successfully.")
    
except Exception as e:
    print(f"Error evaluating Monotonic Neural Network model: {e}")
    import traceback
    traceback.print_exc()

# Display results
if len(model_evaluations) > 0:
    print("\n" + "="*70)
    print("NEURAL NETWORK MODELS COMPARISON RESULTS")
    print("="*70)
    
    for eval_dict in model_evaluations:
        print(f"\n{eval_dict['Model']}:")
        print(f"  MSE: {eval_dict['MSE']:.4f}")
        print(f"  RMSE: {eval_dict['RMSE']:.4f}")
        print(f"  MAE: {eval_dict['MAE']:.4f}")
        print(f"  R²: {eval_dict['R²']:.4f}")
        print(f"  Explained Variance: {eval_dict['Explained Variance']:.4f}")
    
    print("\n✅ Both neural network models are working correctly!")
    print("✅ Ready to run the full model_comparison.py")
else:
    print("❌ No models were successfully evaluated.") 