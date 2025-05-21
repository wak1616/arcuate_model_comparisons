# Arcuate Sweep Prediction Models

A collection of machine learning models to predict arcuate sweep for ophthalmological procedures, with a focus on both accuracy and prediction smoothness.

## Project Overview

This project compares different machine learning approaches to predict arcuate sweep based on patient's ophthalmological measurements, with special attention to the monotonic relationship between treated astigmatism and arcuate sweep.

## Models

The project implements and compares five different models:

1. **Standard XGBoost (Model 1)**: Baseline gradient boosting model.
2. **Random Forest (Model 2)**: Ensemble model using random forest regression.
3. **XGBoost with Monotonicity Constraint (Model 3)**: XGBoost with monotonicity constraint on the Treated_astig feature.
4. **Smooth XGBoost (Model 4)**: XGBoost with fine-tuned hyperparameters for smoother predictions:
   - Increased tree depth (6 instead of 4)
   - Reduced learning rate (0.02 instead of 0.05)
   - Added min_child_weight parameter (3)
   - Increased number of trees (300)
   - Maintained monotonicity constraint on Treated_astig
5. **Neural Network with Sigmoid-Monotonic Constraint (Model 5)**: Custom PyTorch neural network with a hybrid architecture:
   - Monotonic branch with sigmoid-like shape for Treated_astig
   - Standard MLP branch for other features
   - Combined branches for final prediction

## Feature Selection

All models have been optimized to use the following features:
- Age
- Steep_axis_term
- WTW_IOLMaster
- Treated_astig
- Type
- AL
- LASIK?

Note: MeanK_IOLMaster was removed from all models after analysis showed it had minimal impact on prediction performance.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/arcuate_model_comparisons.git
cd arcuate_model_comparisons

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost torch
```

## Usage

Each model can be run independently from its respective directory:

```bash
# Run a specific model
cd models/xgboost
python model1_xgboost.py

# Run comparison of all models
cd models/comparisons
python model_comparison.py
```

## Results

Performance metrics from 5-fold cross-validation:

| Model | R² | RMSE | MAE |
|-------|-----|------|-----|
| XGBoost | 0.9069 ± 0.0267 | 5.1951 ± 0.6816 | 3.2094 ± 0.2227 |
| XGBoost Selective-Monotonic | 0.9111 ± 0.0241 | 5.0787 ± 0.6518 | 3.2111 ± 0.2170 |
| XGBoost Smooth-Monotonic | 0.9105 ± 0.0257 | 5.0916 ± 0.6811 | 3.2357 ± 0.1918 |
| Random Forest | 0.9070 ± 0.0248 | 5.1959 ± 0.6071 | 3.1700 ± 0.1756 |
| Elastic Net | 0.8458 ± 0.0274 | 6.7246 ± 0.6663 | 4.8560 ± 0.3884 |

The XGBoost Selective-Monotonic model (Model 3) achieves the best overall performance while maintaining monotonicity. The XGBoost Smooth-Monotonic model (Model 4) provides slightly smoother predictions with comparable performance.

## Key Findings

- Monotonicity constraints improve model interpretability
- Model smoothness is important for clinical applications
- XGBoost models with monotonicity constraints offer the best balance of accuracy and interpretability
- Removing MeanK_IOLMaster did not impact model performance, indicating it was a redundant feature
- Feature importance analysis consistently shows Treated_astig, Steep_axis_term, and Type as the most influential predictors
- Streamlined models with fewer features are more efficient while maintaining high accuracy

## Recommended Model for Production

After comprehensive evaluation, the **XGBoost Smooth-Monotonic model (Model 4)** is recommended for production use for the following reasons:

1. **Superior smoothness**: The model provides gradual, smooth transitions in predictions as input values change, which is critical for clinical applications where small measurement variations should not result in abrupt changes in arcuate sweep recommendations.

2. **Maintained monotonicity**: The model preserves the essential monotonic relationship between treated astigmatism and arcuate sweep, ensuring clinical validity.

3. **Strong performance metrics**: While the Selective-Monotonic XGBoost has marginally better MSE/RMSE, the Smooth-Monotonic variant has comparable R² (0.9105 vs 0.9111) with the added benefit of prediction smoothness.

4. **Hyperparameter optimization for clinical use**: The model's hyperparameters were specifically tuned to balance accuracy with smoothness:
   - Lower learning rate (0.02 vs 0.05)
   - Higher max_depth (6 vs 4) for finer gradations
   - Added min_child_weight (3) to prevent overly specific splits
   - Increased number of estimators (300 vs 100) for better ensemble averaging

5. **Production-ready implementation**: The model has been implemented with XGBoost's robust serialization format, enabling straightforward deployment across platforms.

The model's combination of performance, smoothness, and clinical interpretability makes it the optimal choice for real-world ophthalmological applications.

## Visualizations

The project includes various visualizations to compare prediction smoothness and model performance across all implemented approaches. 