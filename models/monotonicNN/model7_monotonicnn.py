import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import os

# Initialize encoders and scalers
# Use separate encoders for Eye and Type
type_le = LabelEncoder()
lasik_le = LabelEncoder()
other_scaler = StandardScaler()
monotonic_scaler = StandardScaler()
target_scaler = StandardScaler()

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

# Load dataset
# Get the script's directory and construct path relative to it
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '../../data/datafinal.csv')
df = pd.read_csv(data_path)

# --- Data Splitting ---
print(f"\nSplitting data into Train/Validation sets...")
df_train, df_val = train_test_split(df, test_size=0.2, random_state=42) # 80% train, 20% validation
print(f"Train set size: {len(df_train)}, Validation set size: {len(df_val)}")

# --- Feature Engineering & Preprocessing (Fit on Train, Transform Train/Val) ---

# Setting up features and target for TRAIN set
target_col = 'Arcuate_sweep_total'
y_train_df = df_train[[target_col]]
x_train = df_train['Treated_astig']
x_train_min = x_train.min() # Calculate min based on train set ONLY

other_features = [
    'Age', 'Steep_axis_term', 'WTW_IOLMaster',
    'AL', 'LASIK?',
    'Treatment_astigmatism',
    'Type'
]

# Handle NaN values - Fit on train, transform train & val
wtw_median = df_train['WTW_IOLMaster'].median()
al_median = df_train['AL'].median()
df_train['WTW_IOLMaster'] = df_train['WTW_IOLMaster'].fillna(wtw_median)
df_train['AL'] = df_train['AL'].fillna(al_median)
df_val['WTW_IOLMaster'] = df_val['WTW_IOLMaster'].fillna(wtw_median) # Use train median for val
df_val['AL'] = df_val['AL'].fillna(al_median) # Use train median for val

# Create monotonic features - Train
monotonic_features_dict_train = {
    'constant': np.ones_like(x_train),
    'linear': x_train,
    'logistic_shift_left_1': 1 / (1 + np.exp(-(x_train+1))),
    'logistic_shift_left_0.5': 1 / (1 + np.exp(-(x_train+0.5))),
    'logistic_center': 1 / (1 + np.exp(-x_train)),
    'logarithmic': np.log(x_train - x_train_min + 1), # Use train_min
    'logistic_shift_right_0.5': 1 / (1 + np.exp(-(x_train-0.5))),
    'logistic_shift_right_1': 1 / (1 + np.exp(-(x_train-1))),
    'logistic_shift_right_1.5': 1 / (1 + np.exp(-(x_train-1.5))),
    'logistic_shift_left_1.5': 1 / (1 + np.exp(-(x_train+1.5)))
}
X_monotonic_train = pd.DataFrame(monotonic_features_dict_train)
X_other_train = df_train[other_features].copy()

# Fit and transform encoders - Train
X_other_train['Type'] = type_le.fit_transform(X_other_train['Type'])
X_other_train['LASIK?'] = lasik_le.fit_transform(X_other_train['LASIK?'])

# Get feature orders AFTER encoding/creation
other_features_order = list(X_other_train.columns)
monotonic_feature_order = list(X_monotonic_train.columns)

# Fit and transform scalers - Train
X_other_scaled_train = pd.DataFrame(other_scaler.fit_transform(X_other_train), columns=other_features_order, index=X_other_train.index)
X_monotonic_scaled_train = pd.DataFrame(monotonic_scaler.fit_transform(X_monotonic_train), columns=monotonic_feature_order, index=X_monotonic_train.index)
y_scaled_train = pd.DataFrame(target_scaler.fit_transform(y_train_df.values.reshape(-1, 1)), columns=[target_col], index=y_train_df.index)

# --- Prepare Validation Set ---
y_val_df = df_val[[target_col]]
x_val = df_val['Treated_astig']

# Create monotonic features - Val (use train_min and train order)
monotonic_features_dict_val = {
    'constant': np.ones_like(x_val),
    'linear': x_val,
    'logistic_shift_left_1': 1 / (1 + np.exp(-(x_val+1))),
    'logistic_shift_left_0.5': 1 / (1 + np.exp(-(x_val+0.5))),
    'logistic_center': 1 / (1 + np.exp(-x_val)),
    'logarithmic': np.log(x_val - x_train_min + 1), # Use train_min
    'logistic_shift_right_0.5': 1 / (1 + np.exp(-(x_val-0.5))),
    'logistic_shift_right_1': 1 / (1 + np.exp(-(x_val-1))),
    'logistic_shift_right_1.5': 1 / (1 + np.exp(-(x_val-1.5))),
    'logistic_shift_left_1.5': 1 / (1 + np.exp(-(x_val+1.5)))
}
X_monotonic_val = pd.DataFrame(monotonic_features_dict_val, columns=monotonic_feature_order)
X_other_val = df_val[other_features].copy()

# Transform encoders - Val (use fitted encoders, handle unseen)
for col in ['Type', 'LASIK?']:
    encoder = type_le if col == 'Type' else lasik_le
    X_other_val[col] = X_other_val[col].map(lambda s: s if s in encoder.classes_ else '<unknown>')
    if '<unknown>' not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, '<unknown>')
    X_other_val[col] = encoder.transform(X_other_val[col])

X_other_val = X_other_val[other_features_order] # Ensure order

# Transform scalers - Val (use fitted scalers)
X_other_scaled_val = pd.DataFrame(other_scaler.transform(X_other_val), columns=other_features_order, index=X_other_val.index)
X_monotonic_scaled_val = pd.DataFrame(monotonic_scaler.transform(X_monotonic_val), columns=monotonic_feature_order, index=X_monotonic_val.index)
y_scaled_val = pd.DataFrame(target_scaler.transform(y_val_df.values.reshape(-1, 1)), columns=[target_col], index=y_val_df.index)

# --- Convert to Tensors ---
x_other_tensor_train = torch.FloatTensor(X_other_scaled_train.values)
x_monotonic_tensor_train = torch.FloatTensor(X_monotonic_scaled_train.values)
y_tensor_train = torch.FloatTensor(y_scaled_train.values)

x_other_tensor_val = torch.FloatTensor(X_other_scaled_val.values)
x_monotonic_tensor_val = torch.FloatTensor(X_monotonic_scaled_val.values)
y_tensor_val = torch.FloatTensor(y_scaled_val.values)

# --- Create DataLoaders ---
batch_size = 32
train_dataset = TensorDataset(x_other_tensor_train, x_monotonic_tensor_train, y_tensor_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(x_other_tensor_val, x_monotonic_tensor_val, y_tensor_val)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # No shuffle for validation

# Initialize model
model = SimpleMonotonicNN(len(other_features_order)) # Use length of ordered features
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# --- Training loop with Validation & Early Stopping ---
print("\nTraining model...")
num_epochs = 1000
best_val_loss = float('inf') # Use validation loss for early stopping
patience = 30
patience_counter = 0

for epoch in range(num_epochs):
    # --- Training Phase ---
    model.train()
    epoch_train_loss = 0
    train_batch_count = 0
    for batch_other, batch_monotonic, batch_y in train_dataloader:
        outputs = model(batch_other, batch_monotonic)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_train_loss += loss.item()
        train_batch_count += 1
    avg_train_loss = epoch_train_loss / train_batch_count

    # --- Validation Phase ---
    model.eval()
    epoch_val_loss = 0
    val_batch_count = 0
    with torch.no_grad():
        for batch_other, batch_monotonic, batch_y in val_dataloader:
            outputs = model(batch_other, batch_monotonic)
            loss = criterion(outputs, batch_y)
            epoch_val_loss += loss.item()
            val_batch_count += 1
    avg_val_loss = epoch_val_loss / val_batch_count

    # --- Early Stopping Check & Model Saving ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        print(f'Epoch {epoch+1}: Validation loss improved to {avg_val_loss:.6f}. Saving model...')
        # Save everything in one file like model5
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'other_scaler': other_scaler,
            'monotonic_scaler': monotonic_scaler,
            'target_scaler': target_scaler,
            'type_label_encoder': type_le,
            'lasik_label_encoder': lasik_le,
            'other_features_order': other_features_order,
            'monotonic_feature_order': monotonic_feature_order,
            'x_train_min': x_train_min,
            'model_config': {
                'other_input_dim': len(other_features_order)
            }
        }, 'monotonic_neural_network_model.pt')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f'Early stopping triggered at epoch {epoch+1} due to no improvement in validation loss for {patience} epochs.')
        break

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

# --- Final Evaluation on Validation Set ---
print("\nFINAL MODEL PERFORMANCE (on validation set using best saved weights):")
# Load the best model weights saved during training
try:
    checkpoint = torch.load('monotonic_neural_network_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
except FileNotFoundError:
    print("Warning: monotonic_neural_network_model.pt not found. Evaluating with the last state of the model.")
except Exception as e:
    print(f"Warning: Error loading model weights: {e}. Evaluating with the last state of the model.")

model.eval()
all_val_preds = []
all_val_true = []
with torch.no_grad():
    for batch_other, batch_monotonic, batch_y in val_dataloader:
        outputs_scaled = model(batch_other, batch_monotonic)
        # Inverse transform predictions and true values for this batch
        preds_inv = target_scaler.inverse_transform(outputs_scaled.numpy())
        true_inv = target_scaler.inverse_transform(batch_y.numpy())
        all_val_preds.append(preds_inv)
        all_val_true.append(true_inv)

# Concatenate results from all batches
predictions = np.concatenate(all_val_preds)
predictions = np.maximum(0.0, predictions)  # Ensure non-negative
y_original_val = np.concatenate(all_val_true)

rmse = np.sqrt(mean_squared_error(y_original_val, predictions))
mae = mean_absolute_error(y_original_val, predictions)
r2 = r2_score(y_original_val, predictions)

print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'R² Score: {r2:.4f}')
   

