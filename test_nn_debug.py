import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Load data using the same function as model_comparison.py
def load_data():
    current_file = Path(__file__)
    project_root = current_file.parent  # Go up to the project root
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

# Load data
print("Loading data...")
X, y = load_data()
print(f"Data shape: X={X.shape}, y={y.shape}")
print(f"Features: {list(X.columns)}")

# Test the Neural Network wrapper
try:
    print("\nTesting Neural Network wrapper...")
    
    # Add neural_net to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/neural_net')))
    
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    
    print("Imports successful")
    
    # Define the classes (same as in model_comparison.py)
    class MonotonicLayer(nn.Module):
        def __init__(self, in_features, out_features):
            super(MonotonicLayer, self).__init__()
            self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
            self.bias = nn.Parameter(torch.zeros(out_features))
            
        def forward(self, x):
            positive_weights = torch.relu(self.weight)
            return torch.matmul(x, positive_weights.t()) + self.bias

    class SigmoidMonotonic(nn.Module):
        def __init__(self, scale_factor=1.0):
            super(SigmoidMonotonic, self).__init__()
            self.scale_factor = scale_factor
            
        def forward(self, x):
            return self.scale_factor * torch.sigmoid(x)

    class HybridMonotonicMLP(nn.Module):
        def __init__(self, num_numeric, num_categorical_encoded, monotonic_feature_idx,
                    hidden_size=32, dropout_rate=0.2):
            super(HybridMonotonicMLP, self).__init__()
            
            self.num_numeric = num_numeric
            self.num_categorical_encoded = num_categorical_encoded
            self.monotonic_feature_idx = monotonic_feature_idx
            self.hidden_size = hidden_size
            
            # Monotonic Path
            self.monotonic_path = nn.Sequential(
                MonotonicLayer(1, hidden_size),
                nn.Tanh(),
                MonotonicLayer(hidden_size, hidden_size),
                nn.Tanh(),
                MonotonicLayer(hidden_size, hidden_size // 2),
                SigmoidMonotonic(scale_factor=20.0),
                MonotonicLayer(hidden_size // 2, 1)
            )
            
            # Standard MLP Path
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
            self.combiner = nn.Sequential(
                nn.Linear(2, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 1)
            )
            
        def forward(self, x):
            monotonic_feature = x[:, self.monotonic_feature_idx].unsqueeze(1)
            
            other_features_idx = list(range(self.num_numeric))
            other_features_idx.remove(self.monotonic_feature_idx)
            other_features_idx.extend(range(self.num_numeric, self.num_numeric + self.num_categorical_encoded))
            
            other_features = x[:, other_features_idx]
            
            monotonic_output = self.monotonic_path(monotonic_feature)
            standard_output = self.standard_mlp(other_features)
            
            combined = torch.cat((monotonic_output, standard_output), dim=1)
            output = self.combiner(combined)
            
            return output

    class NeuralNetworkWrapper:
        def __init__(self):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"  Neural Network using device: {self.device}")
            
            self.numeric_features = ['Age', 'Steep_axis_term', 'WTW_IOLMaster', 'Treated_astig', 'AL']
            self.categorical_features = ['Type', 'LASIK?']
            self.monotonic_feature = 'Treated_astig'
            self.monotonic_feature_idx = self.numeric_features.index(self.monotonic_feature)
            
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), self.numeric_features),
                    ('cat', OneHotEncoder(drop='first'), self.categorical_features)
                ])
            
            self.model = None
            
        def fit(self, X, y):
            print(f"  Fitting on data shape: X={X.shape}, y={y.shape}")
            print(f"  X columns: {list(X.columns)}")
            
            # Check if all required features are present
            missing_features = [f for f in self.numeric_features + self.categorical_features if f not in X.columns]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            X_processed = self.preprocessor.fit_transform(X)
            print(f"  Processed shape: {X_processed.shape}")
            
            n_numeric = len(self.numeric_features)
            n_categorical_encoded = X_processed.shape[1] - n_numeric
            print(f"  n_numeric: {n_numeric}, n_categorical_encoded: {n_categorical_encoded}")
            
            X_tensor = torch.FloatTensor(X_processed).to(self.device)
            y_tensor = torch.FloatTensor(y.values.reshape(-1, 1)).to(self.device)
            
            self.model = HybridMonotonicMLP(
                num_numeric=n_numeric,
                num_categorical_encoded=n_categorical_encoded,
                monotonic_feature_idx=self.monotonic_feature_idx,
                hidden_size=64,
                dropout_rate=0.3
            ).to(self.device)
            
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
            
            self.model.train()
            for epoch in range(10):  # Just 10 epochs for testing
                optimizer.zero_grad()
                outputs = self.model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
                if epoch % 5 == 0:
                    print(f"    Epoch {epoch}, Loss: {loss.item():.4f}")
            
            return self
            
        def predict(self, X):
            print(f"  Predicting on data shape: X={X.shape}")
            X_processed = self.preprocessor.transform(X)
            X_tensor = torch.FloatTensor(X_processed).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                pred = self.model(X_tensor).cpu().numpy()
            
            return pred.flatten()
    
    # Test the wrapper
    print("Creating Neural Network wrapper...")
    nn_wrapper = NeuralNetworkWrapper()
    
    # Split data for testing
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    nn_wrapper.fit(X_train, y_train)
    
    print("Making predictions...")
    y_pred = nn_wrapper.predict(X_test)
    
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Test Results: R² = {r2:.4f}, RMSE = {rmse:.4f}")
    print("Neural Network test completed successfully!")
    
except Exception as e:
    print(f"Error in Neural Network test: {e}")
    import traceback
    traceback.print_exc() 