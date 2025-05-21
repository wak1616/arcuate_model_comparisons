# IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set up output directory for saving files
current_dir = Path(__file__).parent
output_dir = current_dir
os.makedirs(output_dir, exist_ok=True)

"""## Part 1: Getting Dataset into Pandas"""

# LOAD MAIN DATASET
datasets_dir = Path(__file__).resolve().parents[2] / "data"  # Project root directory + data folder
df = pd.read_csv(datasets_dir / "datafinal.csv", encoding='utf-8')

# SET RANDOM_STATE AND SHUFFLE THE DATASET ('df')
df = shuffle(df, random_state=42)

# Find out how many entries are nan and in which columns
print(df.isna().sum())

# Remove trailing white spaces for 'Type', 'Sex', 'Eye', 'LASIK?'
df['Type'] = df['Type'].str.strip()
df['Sex'] = df['Sex'].str.strip()
df['Eye'] = df['Eye'].str.strip()
df['LASIK?'] = df['LASIK?'].str.strip()

# Replace any 'Type' entries that = "singe" to "single"
df['Type'] = df['Type'].replace('singe', 'single')

# Look for any outliers and get an overview of dataset
print(df.describe())
print(df.info())

# Setting up features and target
target = ['Arcuate_sweep_total']

# Add the interaction term to the features list
features = [
    'Age', 'Steep_axis_term', 'WTW_IOLMaster',
    'Treated_astig', 'Type',
    'AL', 'LASIK?'
]

# Define numeric and categorical features
numeric_features = ['Age', 'Steep_axis_term', 'WTW_IOLMaster', 
                   'Treated_astig', 'AL']
categorical_features = ['Type', 'LASIK?']

# Extract the feature requiring monotonicity
monotonic_feature = 'Treated_astig'
monotonic_feature_idx = numeric_features.index(monotonic_feature)

# Split dataset into training and testing sets
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""## Part 2: Preprocessing Data"""

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Fit the preprocessor on training data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)
X_full_preprocessed = preprocessor.transform(X)

# Get the dimension of preprocessed features
n_numeric = len(numeric_features)
n_categorical_encoded = X_train_preprocessed.shape[1] - n_numeric

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_preprocessed)
y_train_tensor = torch.FloatTensor(y_train.values)
X_test_tensor = torch.FloatTensor(X_test_preprocessed)
y_test_tensor = torch.FloatTensor(y_test.values)
X_full_tensor = torch.FloatTensor(X_full_preprocessed)
y_full_tensor = torch.FloatTensor(y.values)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
full_dataset = TensorDataset(X_full_tensor, y_full_tensor)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
full_loader = DataLoader(full_dataset, batch_size=batch_size)

"""## Part 3: Building the Neural Network with Monotonicity Constraint"""

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

# New custom activation function for monotonic path
class SigmoidMonotonic(nn.Module):
    """Scaled sigmoid activation that maintains monotonicity"""
    def __init__(self, scale_factor=1.0):
        super(SigmoidMonotonic, self).__init__()
        self.scale_factor = scale_factor
        
    def forward(self, x):
        return self.scale_factor * torch.sigmoid(x)

class HybridMonotonicMLP(nn.Module):
    """
    A hybrid neural network with two branches:
    1. Monotonic branch: Processes Treated_astig with monotonicity constraint and sigmoid-like curve
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
        
        # Monotonic Path (for Treated_astig) - Updated for sigmoid-like behavior
        self.monotonic_path = nn.Sequential(
            MonotonicLayer(1, hidden_size),
            nn.Tanh(),  # Using Tanh to introduce non-linearity while preserving order
            MonotonicLayer(hidden_size, hidden_size),
            nn.Tanh(),  # More non-linearity
            MonotonicLayer(hidden_size, hidden_size // 2),
            SigmoidMonotonic(scale_factor=20.0),  # Scale sigmoid to match output range
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
        
        # Final combination layer with adaptive weights
        self.combiner = nn.Sequential(
            nn.Linear(2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)
        )
        
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

"""## Part 4: Training and Evaluation Functions"""

def train_model(model, train_loader, optimizer, criterion, epochs=100, early_stopping_patience=15):
    """Train the neural network model"""
    model.train()
    history = []
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        history.append(epoch_loss)
        
        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            model.load_state_dict(best_model_state)
            break
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    # Make sure we use the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return history

def evaluate_model(model, data_loader, criterion):
    """Evaluate the model on a given dataset"""
    model.eval()
    running_loss = 0.0
    predictions = []
    true_values = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            
            predictions.append(outputs.numpy())
            true_values.append(targets.numpy())
    
    predictions = np.vstack(predictions).flatten()
    true_values = np.vstack(true_values).flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values, predictions)
    
    return {
        'loss': running_loss / len(data_loader.dataset),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'predictions': predictions,
        'true_values': true_values
    }

def k_fold_cv(X, y, n_splits=5, hidden_size=32, dropout_rate=0.2, 
             lr=0.001, weight_decay=1e-5, epochs=100):
    """Perform k-fold cross-validation"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"Fold {fold}/{n_splits}")
        
        # Split data
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train_fold), torch.FloatTensor(y_train_fold))
        val_dataset = TensorDataset(torch.FloatTensor(X_val_fold), torch.FloatTensor(y_val_fold))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Create model
        model = HybridMonotonicMLP(
            num_numeric=n_numeric,
            num_categorical_encoded=n_categorical_encoded,
            monotonic_feature_idx=monotonic_feature_idx,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate
        )
        
        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Train model
        train_model(model, train_loader, optimizer, criterion, epochs=epochs)
        
        # Evaluate
        val_results = evaluate_model(model, val_loader, criterion)
        fold_results.append(val_results)
        
        print(f"Fold {fold} - MAE: {val_results['mae']:.4f}, RMSE: {val_results['rmse']:.4f}, R2: {val_results['r2']:.4f}")
    
    # Calculate average metrics
    avg_mae = np.mean([r['mae'] for r in fold_results])
    avg_rmse = np.mean([r['rmse'] for r in fold_results])
    avg_r2 = np.mean([r['r2'] for r in fold_results])
    std_mae = np.std([r['mae'] for r in fold_results])
    std_rmse = np.std([r['rmse'] for r in fold_results])
    std_r2 = np.std([r['r2'] for r in fold_results])
    
    print("\nAverage scores across all folds:")
    print(f"MAE: {avg_mae:.4f} (+/- {std_mae:.4f})")
    print(f"RMSE: {avg_rmse:.4f} (+/- {std_rmse:.4f})")
    print(f"R2: {avg_r2:.4f} (+/- {std_r2:.4f})")
    
    return fold_results, (avg_mae, avg_rmse, avg_r2, std_mae, std_rmse, std_r2)

"""## Part 5: Model Training and Evaluation"""

print("Starting cross-validation...")
# Perform cross-validation
cv_results, cv_metrics = k_fold_cv(
    X_full_preprocessed, y.values, 
    n_splits=5,
    hidden_size=64,
    dropout_rate=0.3,
    lr=0.001,
    weight_decay=1e-5,
    epochs=150
)
avg_mae, avg_rmse, avg_r2, std_mae, std_rmse, std_r2 = cv_metrics

# Train final model on all data
print("\nTraining final model on full dataset...")
model = HybridMonotonicMLP(
    num_numeric=n_numeric,
    num_categorical_encoded=n_categorical_encoded,
    monotonic_feature_idx=monotonic_feature_idx,
    hidden_size=64,
    dropout_rate=0.3
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
train_history = train_model(model, train_loader, optimizer, criterion, epochs=150)

# Evaluate on test set
test_results = evaluate_model(model, test_loader, criterion)
print("\nTEST DATASET METRICS:")
print(f"Mean Absolute Error: {test_results['mae']:.4f}")
print(f"Root Mean Squared Error: {test_results['rmse']:.4f}")
print(f"R^2 Score: {test_results['r2']:.4f}")

# Evaluate on full dataset
full_results = evaluate_model(model, full_loader, criterion)
print("\nFULL DATASET METRICS:")
print(f"Mean Absolute Error: {full_results['mae']:.4f}")
print(f"Root Mean Squared Error: {full_results['rmse']:.4f}")
print(f"R^2 Score: {full_results['r2']:.4f}")

"""## Part 6: Visualizations and Analysis"""

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(train_history)
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.savefig(output_dir / 'model5_training_history.png')
plt.close()

# Visualize Predictions vs Actual Values
plt.figure(figsize=(10, 10))
plt.scatter(test_results['true_values'], test_results['predictions'], alpha=0.5)
plt.xlabel('Actual Arcuate Sweep')
plt.ylabel('Predicted Arcuate Sweep')
plt.title('Neural Network: Arcuate Sweep Prediction with Monotonicity Constraint')
plt.plot([10, 55], [10, 55], 'r--', alpha=0.5)
plt.savefig(output_dir / 'model5_predictions_vs_actual.png')
plt.close()

# Plot residuals
residuals = test_results['predictions'] - test_results['true_values']
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Neural Network Prediction Error')
plt.savefig(output_dir / 'model5_residuals_distribution.png')
plt.close()

"""## Part 7: Visualize the Monotonic Relationship"""

# Function to visualize monotonicity by varying Treated_astig while keeping other features fixed
def visualize_monotonicity(model, X_example, preprocessor, monotonic_feature_idx, monotonic_feature_name):
    model.eval()
    
    # Create a range of values for Treated_astig
    astig_min = X[monotonic_feature_name].min()
    astig_max = X[monotonic_feature_name].max()
    astig_values = np.linspace(astig_min, astig_max, 100)
    
    predictions = []
    
    # Create a copy of the sample data
    sample_data = X_example.copy()
    
    with torch.no_grad():
        for astig in astig_values:
            # Update the Treated_astig value
            sample_data[monotonic_feature_name] = astig
            
            # Preprocess
            processed = preprocessor.transform(sample_data)
            processed_tensor = torch.FloatTensor(processed)
            
            # Make prediction
            output = model(processed_tensor)
            predictions.append(output.numpy().item())
    
    # Plot the relationship
    plt.figure(figsize=(10, 6))
    plt.plot(astig_values, predictions, 'b-', linewidth=2)
    plt.xlabel('Treated Astigmatism (D)')
    plt.ylabel('Predicted Arcuate Sweep')
    plt.title('Neural Network: Relationship Between Treated Astigmatism and Predicted Arcuate Sweep\n(with Sigmoid-Monotonic Constraint)')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'model5_monotonicity_visualization.png')
    plt.close()
    
    return astig_values, predictions

# Select a single record for visualization instead of trying to compute the mean
# This avoids issues with categorical variables
X_example = X.iloc[[0]].copy()  # Use the first record as an example

# Visualize the monotonic relationship
print("\nVisualizing monotonic relationship...")
astig_values, nn_predictions = visualize_monotonicity(
    model, X_example, preprocessor, monotonic_feature_idx, monotonic_feature
)

# Compare with previous model (model3) if available
try:
    # Try to load the previous model (XGBoost) predictions
    xgboost_model_path = Path(__file__).resolve().parents[1] / "xgboost_monotonic" / "XGBoost_monotonic_model_latest.json"
    if xgboost_model_path.exists():
        print("\nComparing with XGBoost monotonic model...")
        import xgboost as xgb
        
        # Load the XGBoost model
        xgb_model = xgb.Booster()
        xgb_model.load_model(str(xgboost_model_path))
        
        # Get XGBoost predictions for the same astig values
        xgb_predictions = []
        for astig in astig_values:
            X_temp = X_example.copy()
            # Convert categorical columns to proper category dtype
            for cat_col in categorical_features:
                X_temp[cat_col] = X_temp[cat_col].astype('category')
            X_temp[monotonic_feature] = astig
            # Use enable_categorical=True for categorical features
            dmatrix = xgb.DMatrix(X_temp, enable_categorical=True)
            pred = xgb_model.predict(dmatrix)
            xgb_predictions.append(pred.mean())
        
        # Try to load the smoother XGBoost model (model4) if available
        smooth_model_path = Path(__file__).resolve().parents[1] / "xgboost_smooth" / "XGBoost_smooth_model_latest.json"
        if smooth_model_path.exists():
            smooth_xgb_model = xgb.Booster()
            smooth_xgb_model.load_model(str(smooth_model_path))
            
            smooth_predictions = []
            for astig in astig_values:
                X_temp = X_example.copy()
                # Convert categorical columns to proper category dtype
                for cat_col in categorical_features:
                    X_temp[cat_col] = X_temp[cat_col].astype('category')
                X_temp[monotonic_feature] = astig
                # Use enable_categorical=True for categorical features
                dmatrix = xgb.DMatrix(X_temp, enable_categorical=True)
                pred = smooth_xgb_model.predict(dmatrix)
                smooth_predictions.append(pred.mean())
                
            has_smooth_model = True
        else:
            has_smooth_model = False
        
        # Plot comparison
        plt.figure(figsize=(12, 7))
        plt.plot(astig_values, nn_predictions, 'b-', linewidth=2, label='Neural Network (Model 5 - Sigmoid)')
        plt.plot(astig_values, xgb_predictions, 'g--', linewidth=2, label='XGBoost Monotonic (Model 3)')
        
        if has_smooth_model:
            plt.plot(astig_values, smooth_predictions, 'r-.', linewidth=2, label='Smooth XGBoost (Model 4)')
        
        plt.xlabel('Treated Astigmatism (D)')
        plt.ylabel('Predicted Arcuate Sweep')
        plt.title('Comparison of Models with Monotonicity Constraints')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(output_dir / 'model_comparison_nn_vs_xgboost.png')
        plt.close()
        print("Comparison plot saved")
    else:
        print("Previous model file not found for comparison.")
except Exception as e:
    print(f"Error comparing with previous model: {e}")
    
    # Still create a visualization of the neural network predictions alone
    plt.figure(figsize=(10, 6))
    plt.plot(astig_values, nn_predictions, 'b-', linewidth=2, label='Neural Network (Model 5 - Sigmoid)')
    plt.xlabel('Treated Astigmatism (D)')
    plt.ylabel('Predicted Arcuate Sweep')
    plt.title('Neural Network: Monotonic Relationship Between\nTreated Astigmatism and Predicted Arcuate Sweep')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'model5_monotonicity_visualization.png')
    plt.close()
    print("Created neural network visualization without comparison")

# Save the model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'preprocessor': preprocessor,
    'model_config': {
        'num_numeric': n_numeric,
        'num_categorical_encoded': n_categorical_encoded,
        'monotonic_feature_idx': monotonic_feature_idx,
        'hidden_size': 64,
        'dropout_rate': 0.3
    }
}, output_dir / 'neural_network_model.pt')
print(f"\nModel saved as '{output_dir / 'neural_network_model.pt'}'")

# Create a summary table for quick reference
print("\n" + "="*60)
print("NEURAL NETWORK WITH SIGMOID-MONOTONICITY CONSTRAINT (MODEL 5)")
print("="*60)
print("\nThis model uses a PyTorch neural network with a hybrid architecture:")
print("• Monotonic branch: Processes Treated_astig with sigmoid-like curve and guaranteed monotonicity")
print("• Standard MLP branch: Processes all other features with flexible interactions")
print("• The branches are combined for the final prediction")

print("\nModel Performance:")
print(f"Cross-Validation Results:")
print(f"MAE: {avg_mae:.4f} (±{std_mae:.4f})")
print(f"RMSE: {avg_rmse:.4f} (±{std_rmse:.4f})")
print(f"R²: {avg_r2:.4f} (±{std_r2:.4f})")

print(f"\nTest Dataset:")
print(f"MAE: {test_results['mae']:.4f}")
print(f"RMSE: {test_results['rmse']:.4f}")
print(f"R²: {test_results['r2']:.4f}")

print(f"\nFull Dataset:")
print(f"MAE: {full_results['mae']:.4f}")
print(f"RMSE: {full_results['rmse']:.4f}")
print(f"R²: {full_results['r2']:.4f}")

print(f"\nPlots and visualizations saved to: {output_dir}") 