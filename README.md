# Arcuate Sweep Prediction Models

A comprehensive machine learning project comparing different approaches to predict arcuate sweep for ophthalmological procedures, with emphasis on accuracy, monotonicity constraints, and clinical interpretability.

## Project Overview

This project implements and compares seven different machine learning models to predict arcuate sweep based on patient ophthalmological measurements. The models range from traditional ensemble methods to advanced neural networks with custom monotonicity constraints, providing a thorough evaluation of different approaches for clinical prediction tasks.

## Models

### 1. **XGBoost (Model 1)** - `models/xgboost/`
- **Type**: Gradient boosting baseline model
- **Features**: Standard XGBoost with optimized hyperparameters
- **Performance**: R² = 0.9069 ± 0.0267, RMSE = 5.1951 ± 0.6816
- **Use Case**: High-performance baseline without constraints

### 2. **Random Forest (Model 2)** - `models/random_forest/`
- **Type**: Ensemble regression model
- **Features**: Optimized Random Forest with hyperparameter tuning
- **Performance**: R² = 0.9070 ± 0.0248, RMSE = 5.1959 ± 0.6071
- **Use Case**: Robust ensemble method with feature importance analysis

### 3. **XGBoost Selective-Monotonic (Model 3)** - `models/xgboost_monotonic/`
- **Type**: Gradient boosting with monotonicity constraints
- **Features**: XGBoost with monotonic constraint on `Treated_astig` feature
- **Performance**: R² = 0.9101 ± 0.0250, RMSE = 5.1056 ± 0.6724
- **Use Case**: Clinical interpretability with guaranteed monotonic relationship

### 4. **XGBoost Smooth-Monotonic (Model 4)** - `models/xgboost_smooth/`
- **Type**: Fine-tuned gradient boosting for smooth predictions
- **Features**: 
  - Increased tree depth (6 vs 4)
  - Reduced learning rate (0.02 vs 0.05)
  - Added min_child_weight (3) for smoother splits
  - Maintained monotonicity on `Treated_astig`
- **Performance**: R² = 0.9101 ± 0.0268, RMSE = 5.0988 ± 0.7110
- **Use Case**: Smooth predictions for clinical applications requiring gradual transitions

### 5. **Elastic Net (Model 4)** - `models/elastic_net/`
- **Type**: Regularized linear regression
- **Features**: L1/L2 regularization with hyperparameter optimization
- **Performance**: R² = 0.8458 ± 0.0274, RMSE = 6.7246 ± 0.6663
- **Use Case**: Linear baseline with feature selection capabilities

### 6. **Neural Network (Model 5)** - `models/neural_net/`
- **Type**: Hybrid neural network with monotonicity constraints
- **Features**:
  - Dual-branch architecture (monotonic + standard MLP)
  - Custom monotonic layers with positive weights
  - Sigmoid-monotonic activation for `Treated_astig`
  - Combines branches for final prediction
- **Performance**: R² = 0.8970 ± 0.0183, RMSE = 5.4953 ± 0.5520
- **Use Case**: Deep learning approach with enforced monotonicity

### 7. **Monotonic Neural Network (Model 7)** - `models/monotonicNN/`
- **Type**: Specialized monotonic neural network
- **Features**:
  - Simple monotonic architecture
  - Multiple monotonic transformations of `Treated_astig`
  - Unconstrained path for other features
- **Performance**: R² = 0.8576 ± 0.0218, RMSE = 6.4585 ± 0.4011
- **Use Case**: Strict monotonicity enforcement with neural network flexibility

## Features Used

All models use the following standardized feature set:

### Numeric Features:
- **Age**: Patient age
- **Steep_axis_term**: Steep axis measurement (trigonometric transformation)
- **WTW_IOLMaster**: White-to-white measurement from IOLMaster
- **Treated_astig**: Treated astigmatism (primary monotonic feature)
- **AL**: Axial length

### Categorical Features:
- **Type**: Procedure type (single/paired)
- **LASIK?**: Previous LASIK surgery (no/myopic/hyperopic)

**Note**: `MeanK_IOLMaster` was removed after analysis showed minimal impact on prediction performance across all models.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd arcuate_model_comparisons

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost torch joblib
```

## Usage

### Running Individual Models

Each model can be executed independently from its respective directory:

```bash
# XGBoost models
cd models/xgboost && python model1_xgboost.py
cd models/xgboost_monotonic && python model3_xgboost_monotonic.py
cd models/xgboost_smooth && python model4_xgboost_smooth.py

# Other models
cd models/random_forest && python model2_random_forest.py
cd models/elastic_net && python model4_elastic_net.py
cd models/neural_net && python model5_neural_net.py
cd models/monotonicNN && python model7_monotonicnn.py
```

### Running Model Comparison

```bash
# Compare all models with cross-validation
cd models/comparisons && python model_comparison.py

# Display formatted results
python display_results.py
```

## Performance Results

### Cross-Validation Performance (5-fold)

| Model | R² Score | RMSE | MAE | Stability |
|-------|----------|------|-----|-----------|
| **XGBoost Selective-Monotonic** | 0.9101 ± 0.0250 | 5.1056 ± 0.6724 | 3.2249 ± 0.2059 | High |
| **XGBoost Smooth-Monotonic** | 0.9101 ± 0.0268 | 5.0988 ± 0.7110 | 3.2193 ± 0.2107 | High |
| **Random Forest** | 0.9070 ± 0.0248 | 5.1959 ± 0.6071 | 3.1700 ± 0.1756 | High |
| **XGBoost** | 0.9069 ± 0.0267 | 5.1951 ± 0.6816 | 3.2094 ± 0.2227 | High |
| **Neural Network** | 0.8970 ± 0.0183 | 5.4953 ± 0.5520 | 3.8090 ± 0.4168 | Medium |
| **Monotonic Neural Network** | 0.8576 ± 0.0218 | 6.4585 ± 0.4011 | 4.7082 ± 0.2078 | Medium |
| **Elastic Net** | 0.8458 ± 0.0274 | 6.7246 ± 0.6663 | 4.8560 ± 0.3884 | Medium |

## Key Findings

### Model Performance
- **Top Performers**: XGBoost models (both monotonic variants) achieve the highest R² scores (~0.91)
- **Monotonicity Trade-off**: Minimal performance loss when enforcing monotonicity constraints
- **Smoothness vs Accuracy**: Model 4 (Smooth-Monotonic) provides comparable accuracy with improved prediction smoothness

### Feature Importance
Consistent across all models:
1. **Treated_astig**: Primary predictor (monotonic relationship enforced where applicable)
2. **Steep_axis_term**: Secondary geometric factor
3. **Age**: Patient demographic factor
4. **AL (Axial Length)**: Anatomical measurement
5. **WTW_IOLMaster**: Corneal measurement

### Clinical Insights
- **Monotonicity**: Essential for clinical trust - higher treated astigmatism should predict larger arcuate sweep
- **Smoothness**: Important for clinical applications where small measurement variations shouldn't cause abrupt prediction changes
- **Interpretability**: Tree-based models provide better feature importance analysis than neural networks

## Model Selection Guidelines

### For Maximum Accuracy:
- **XGBoost Selective-Monotonic (Model 3)** or **XGBoost Smooth-Monotonic (Model 4)**

### For Clinical Interpretability:
- **XGBoost Selective-Monotonic (Model 3)** - Best balance of performance and interpretability

### For Smooth Predictions:
- **XGBoost Smooth-Monotonic (Model 4)** - Optimized for gradual transitions

### For Research/Experimentation:
- **Neural Network (Model 5)** - Custom architecture with monotonic constraints
- **Monotonic Neural Network (Model 7)** - Specialized monotonic transformations

## File Organization

```
arcuate_model_comparisons/
├── data/
│   └── datafinal.csv
├── models/
│   ├── xgboost/                    # Model 1
│   ├── random_forest/              # Model 2  
│   ├── xgboost_monotonic/          # Model 3
│   ├── xgboost_smooth/             # Model 4 (XGBoost)
│   ├── elastic_net/                # Model 4 (Elastic Net)
│   ├── neural_net/                 # Model 5
│   ├── monotonicNN/                # Model 7
│   └── comparisons/                # Model comparison tools
└── README.md
```

Each model directory contains:
- Main model script (`model*.py`)
- Saved model files (`.json`, `.pkl`, `.pth`)
- Generated visualizations (`.png`)
- Performance metrics and results

## Visualizations

The project generates comprehensive visualizations including:
- **Prediction vs Actual scatter plots**
- **Residual distributions**
- **Feature importance charts**
- **Monotonicity verification plots**
- **Model comparison charts**
- **Cross-validation stability analysis**

## Technical Notes

- **Reproducibility**: All models use `random_state=42` for consistent results
- **Cross-Validation**: 5-fold CV with stratified sampling
- **Early Stopping**: Implemented to prevent overfitting
- **Hyperparameter Optimization**: Grid search and Bayesian optimization used
- **File Organization**: All outputs saved in respective model subdirectories
- **Data Preprocessing**: Standardized across all models with proper train/test splits

## Contributing

When adding new models:
1. Create a new directory under `models/`
2. Follow the naming convention `model*_description.py`
3. Save all outputs in the model's subdirectory
4. Update the comparison script to include the new model
5. Document the model's unique features and use cases 