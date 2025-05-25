import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Get current directory and load the saved model
current_dir = Path(__file__).parent
model_path = current_dir / "elastic_net_best_model.pkl"
model = joblib.load(model_path)

# Get the feature names
numeric_features = ['Age', 'Steep_axis_term', 'WTW_IOLMaster', 'Treated_astig', 'AL']
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

# Sort by absolute coefficient value
feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

# Print feature importance
print("\nFeature Importance (by coefficient magnitude):")
print(feature_importance[['Feature', 'Coefficient', 'Abs_Coefficient']].to_string(index=False))

# Save feature importance to file
feature_importance.to_csv(current_dir / 'elastic_net_feature_importance.csv', index=False)
print(f"\nFeature importance saved to {current_dir / 'elastic_net_feature_importance.csv'}")

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
plt.title('Elastic Net Coefficients')
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(current_dir / 'elastic_net_coefficients.png')
plt.show() 