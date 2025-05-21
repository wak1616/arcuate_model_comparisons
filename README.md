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
| XGBoost | 0.9064 ± 0.0272 | 5.2076 ± 0.6939 | 3.2175 ± 0.2161 |
| XGBoost Selective-Monotonic | 0.9105 ± 0.0239 | 5.0995 ± 0.6552 | 3.2481 ± 0.2297 |
| XGBoost Smooth-Monotonic | 0.9109 ± 0.0237 | 5.0861 ± 0.6447 | 3.2446 ± 0.1858 |
| Random Forest | 0.9064 ± 0.0234 | 5.2190 ± 0.5728 | 3.1756 ± 0.1717 |
| Elastic Net | 0.8452 ± 0.0279 | 6.7376 ± 0.6788 | 4.8654 ± 0.3939 |
| Neural Network Sigmoid-Monotonic | 0.8885 ± 0.0210 | 5.6882 ± 0.3511 | 3.9088 ± 0.2824 |

The XGBoost Smooth-Monotonic model (Model 4) achieves the best overall performance while maintaining monotonicity. The Neural Network with Sigmoid-Monotonic constraint (Model 5) provides the most biologically plausible sigmoid-shaped relationship between treated astigmatism and arcuate sweep.

## Key Findings

- Monotonicity constraints improve model interpretability
- Model smoothness is important for clinical applications
- The sigmoid-shaped relationship produced by the neural network matches clinical expectations
- XGBoost models offer slightly better accuracy while neural networks provide more natural curves

## Visualizations

The project includes various visualizations to compare prediction smoothness and model performance across all implemented approaches. 