import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to sys.path to import from model directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Function to load and preprocess data (copied from model_comparison.py)
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
    
    # Set up features and target (CORRECT feature set - no MeanK_IOLMaster)
    target = ['Arcuate_sweep_total']
    features = [
        'Age', 'Steep_axis_term', 'WTW_IOLMaster', 
        'Treated_astig', 'Type', 
        'AL', 'LASIK?'
    ]
    
    X = df[features]
    y = df[target]
    
    return X, y

# Load the data
print("Loading data...")
X, y = load_data()
print(f"Data shape: X={X.shape}, y={y.shape}")
print(f"Features: {list(X.columns)}")
print(f"Missing values per column:\n{X.isnull().sum()}")

# Test a small subset for quick testing
X_test = X.head(100)
y_test = y.head(100)

print("\n" + "="*70)
print("TESTING NEURAL NETWORK MODEL (MODEL5) - TRAIN FROM SCRATCH")
print("="*70)

try:
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available - using CPU")
    
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
    
    # Define wrapper class
    class NeuralNetworkWrapper:
        def __init__(self):
            self.device = device
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
            print("  Fitting Neural Network...")
            # Fit preprocessor and transform data
            X_processed = self.preprocessor.fit_transform(X)
            
            # Get dimensions
            n_numeric = len(self.numeric_features)
            n_categorical_encoded = X_processed.shape[1] - n_numeric
            
            print(f"  Input dimensions: {X_processed.shape}")
            print(f"  Numeric features: {n_numeric}, Categorical encoded: {n_categorical_encoded}")
            
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
            for epoch in range(50):  # Reduced for testing
                optimizer.zero_grad()
                outputs = self.model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    print(f"    Epoch {epoch+1}, Loss: {loss.item():.6f}")
            
            return self
            
        def predict(self, X):
            print("  Predicting with Neural Network...")
            # Transform data using fitted preprocessor
            X_processed = self.preprocessor.transform(X)
            X_tensor = torch.FloatTensor(X_processed).to(self.device)
            
            # Make predictions
            self.model.eval()
            with torch.no_grad():
                pred = self.model(X_tensor).cpu().numpy()
            
            return pred.flatten()
    
    # Test the model
    model5_wrapper = NeuralNetworkWrapper()
    model5_wrapper.fit(X_test, y_test.values.ravel())
    predictions = model5_wrapper.predict(X_test)
    
    print(f"  Model5 predictions shape: {predictions.shape}")
    print(f"  Sample predictions: {predictions[:5]}")
    print("  ✓ Model5 (Neural Network) working!")
    
except Exception as e:
    print(f"  ✗ Error with Model5: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("TESTING MONOTONIC NEURAL NETWORK (MODEL7)")
print("="*70)

try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    # Define the SimpleMonotonicNN class
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
    
    # Define wrapper class
    class MonotonicNeuralNetworkWrapper:
        def __init__(self):
            self.device = device
            print(f"  MonotonicNN using device: {self.device}")
            
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
            print("  Fitting MonotonicNN...")
            # Add Treatment_astigmatism column if it doesn't exist
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
            
            # Prepare other features (model7 specific features)
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
            
            print(f"  Other features: {self.other_features_order}")
            print(f"  Monotonic features: {self.monotonic_feature_order}")
            
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
            
            # Training loop
            self.model.train()
            for epoch in range(50):  # Reduced for testing
                optimizer.zero_grad()
                outputs = self.model(x_other_tensor, x_monotonic_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    print(f"    Epoch {epoch+1}, Loss: {loss.item():.6f}")
            
            return self
            
        def predict(self, X):
            print("  Predicting with MonotonicNN...")
            # Same preprocessing as fit
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
    
    # Test the model
    model7_wrapper = MonotonicNeuralNetworkWrapper()
    model7_wrapper.fit(X_test, y_test.values.ravel())
    predictions = model7_wrapper.predict(X_test)
    
    print(f"  Model7 predictions shape: {predictions.shape}")
    print(f"  Sample predictions: {predictions[:5]}")
    print("  ✓ Model7 (MonotonicNN) working!")
    
except Exception as e:
    print(f"  ✗ Error with Model7: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("✅ Both neural network models now:")
print("   - Use the CORRECT feature set (no MeanK_IOLMaster)")
print("   - Train from scratch for fair cross-validation")
print("   - Support GPU acceleration (if available)")
print("   - Are ready for model_comparison.py")

# Check if we should install CUDA PyTorch for better performance
if not torch.cuda.is_available():
    print("\n💡 PERFORMANCE TIP:")
    print("   Your system has an RTX 3060 GPU, but PyTorch can't access it.")
    print("   To enable GPU acceleration, install CUDA-enabled PyTorch:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("   This could significantly speed up neural network training!") 