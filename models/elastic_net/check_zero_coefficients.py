import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Load the saved model
model_path = Path("elastic_net_best_model.pkl")
model = joblib.load(model_path)

# Get the feature names
numeric_features = ['Age', 'Steep_axis_term', 'WTW_IOLMaster', 'MeanK_IOLMaster', 'Treated_astig', 'AL']
categorical_features = ['Type', 'LASIK?']

# Get feature names from the pipeline
feature_names = (
    numeric_features +
    list(model.named_steps['preprocessor']
         .named_transformers_['cat']
         .get_feature_names_out(categorical_features))
)

# Get coefficients
coefficients = model.named_steps['regressor'].coef_

# Create a DataFrame for feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})

# Check for zero coefficients
zero_coeffs = feature_importance[feature_importance['Coefficient'] == 0]
print("\nFeatures with EXACTLY zero coefficients (completely dropped):")
if zero_coeffs.empty:
    print("No features were completely dropped (all have non-zero coefficients)")
else:
    print(zero_coeffs[['Feature', 'Coefficient']].to_string(index=False))

# Check for very small coefficients (effectively dropped)
small_threshold = 0.01  # A small threshold value
small_coeffs = feature_importance[abs(feature_importance['Coefficient']) < small_threshold]
print("\nFeatures with very small coefficients (effectively negligible):")
if small_coeffs.empty:
    print("No features have coefficients smaller than", small_threshold)
else:
    print(small_coeffs[['Feature', 'Coefficient']].to_string(index=False))

# Display all coefficients for reference
print("\nAll feature coefficients (ordered by magnitude):")
feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
print(feature_importance[['Feature', 'Coefficient']].to_string(index=False))

# Capture regularization parameters
alpha = model.named_steps['regressor'].alpha
l1_ratio = model.named_steps['regressor'].l1_ratio

print(f"\nRegularization parameters:")
print(f"Alpha (overall regularization strength): {alpha}")
print(f"L1 ratio (LASSO vs Ridge mix): {l1_ratio}")
print(f"Note: L1 ratio of 1.0 is pure LASSO, 0.0 is pure Ridge") 