import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.compose import ColumnTransformer
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to sys.path to be consistent with other scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Set up output directory for saving files
current_dir = Path(__file__).parent
output_dir = current_dir
os.makedirs(output_dir, exist_ok=True)

def load_data():
    """Load and preprocess the data, similar to other models"""
    # Get absolute path to the data file
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent  # Go up two levels to the project root
    data_file = project_root / "data" / "datafinal.csv"
    
    df = pd.read_csv(data_file, encoding='utf-8')
    
    # Basic preprocessing
    df['Type'] = df['Type'].str.strip()
    df['Sex'] = df['Sex'].str.strip()
    df['Eye'] = df['Eye'].str.strip()
    df['LASIK?'] = df['LASIK?'].str.strip()
    df['Type'] = df['Type'].replace('singe', 'single')
    
    # Return the full dataframe for more detailed analysis
    return df

def calculate_vif(X):
    """Calculate Variance Inflation Factor for each feature"""
    # VIF dataframe
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    
    # Calculate VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    return vif_data.sort_values("VIF", ascending=False)

def plot_correlation_matrix(df, numeric_features, title="Correlation Matrix of Numeric Features"):
    """Create a correlation matrix heatmap for numeric features"""
    corr = df[numeric_features].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=True, fmt=".2f")
    
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    plot_file = output_dir / 'correlation_matrix.png'
    plt.savefig(plot_file)
    print(f"Correlation matrix visualization saved to '{plot_file}'")
    plt.close()
    
    return corr

def calculate_feature_correlations_with_target(df, numeric_features, target):
    """Calculate correlations between features and the target variable"""
    correlations = []
    
    for feature in numeric_features:
        pearson_corr, pearson_p = pearsonr(df[feature], df[target])
        spearman_corr, spearman_p = spearmanr(df[feature], df[target])
        
        correlations.append({
            'Feature': feature,
            'Pearson_Correlation': pearson_corr,
            'Pearson_P_Value': pearson_p,
            'Spearman_Correlation': spearman_corr,
            'Spearman_P_Value': spearman_p
        })
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('Pearson_Correlation', ascending=False)
    
    return corr_df

def plot_feature_target_correlations(correlation_df):
    """Create a bar plot of feature correlations with target"""
    plt.figure(figsize=(12, 8))
    
    x = correlation_df['Feature']
    y1 = correlation_df['Pearson_Correlation']
    y2 = correlation_df['Spearman_Correlation']
    
    bar_width = 0.35
    index = np.arange(len(x))
    
    bar1 = plt.bar(index, y1, bar_width, label='Pearson Correlation')
    bar2 = plt.bar(index + bar_width, y2, bar_width, label='Spearman Correlation')
    
    plt.xlabel('Features')
    plt.ylabel('Correlation Coefficient')
    plt.title('Feature Correlations with Arcuate Sweep')
    plt.xticks(index + bar_width/2, x, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plot_file = output_dir / 'feature_target_correlations.png'
    plt.savefig(plot_file)
    print(f"Feature-target correlation visualization saved to '{plot_file}'")
    plt.close()

def plot_feature_relationships(df, numeric_features, target='Arcuate_sweep_total'):
    """Create scatter plots for each feature vs target"""
    num_features = len(numeric_features)
    num_cols = 2
    num_rows = (num_features + 1) // 2
    
    plt.figure(figsize=(15, 4 * num_rows))
    
    for i, feature in enumerate(numeric_features):
        plt.subplot(num_rows, num_cols, i + 1)
        
        # Plot scatter with regression line
        sns.regplot(x=feature, y=target, data=df, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
        
        plt.grid(True, alpha=0.3)
        plt.title(f"{feature} vs {target}")
    
    plt.tight_layout()
    plot_file = output_dir / 'feature_target_relationships.png'
    plt.savefig(plot_file)
    print(f"Feature relationship visualizations saved to '{plot_file}'")
    plt.close()

def analyze_categorical_features(df, categorical_features, target='Arcuate_sweep_total'):
    """Analyze the impact of categorical features on the target"""
    plt.figure(figsize=(14, 5 * len(categorical_features)))
    
    for i, feature in enumerate(categorical_features):
        plt.subplot(len(categorical_features), 1, i + 1)
        
        # Calculate mean and std of target for each category
        grouped = df.groupby(feature)[target].agg(['mean', 'std', 'count']).reset_index()
        grouped = grouped.sort_values('mean', ascending=False)
        
        # Create bar plot
        bars = plt.bar(grouped[feature], grouped['mean'], yerr=grouped['std'], capsize=10, 
                        color='lightblue', edgecolor='black')
        
        # Add count as text on each bar
        for bar, count in zip(bars, grouped['count']):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'n={count}',
                    ha='center', va='bottom', rotation=0)
        
        plt.title(f'Impact of {feature} on {target}')
        plt.xlabel(feature)
        plt.ylabel(f'Mean {target}')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
    plt.tight_layout()
    plot_file = output_dir / 'categorical_feature_impact.png'
    plt.savefig(plot_file)
    print(f"Categorical feature impact visualization saved to '{plot_file}'")
    plt.close()

def main():
    print("=== Multicollinearity Analysis for Arcuate Sweep Prediction ===\n")
    
    # Load data
    print("Loading and preprocessing data...")
    df = load_data()
    
    # Define features and target
    target = 'Arcuate_sweep_total'
    numeric_features = ['Age', 'Steep_axis_term', 'WTW_IOLMaster', 
                       'Treated_astig', 'AL']
    categorical_features = ['Type', 'LASIK?']
    all_features = numeric_features + categorical_features
    
    print(f"\nAnalyzing {len(all_features)} features: {', '.join(all_features)}")
    print(f"Target variable: {target}")
    
    # Basic stats about the dataset
    print(f"\nDataset contains {df.shape[0]} observations")
    print("\nSummary statistics for numeric features:")
    print(df[numeric_features].describe())
    
    # Calculate and display correlation matrix for numeric features
    print("\n1. Analyzing feature-feature correlations...")
    corr_matrix = plot_correlation_matrix(df, numeric_features)
    
    # Identify highly correlated features
    high_corr_threshold = 0.5
    high_corr_pairs = []
    
    for i in range(len(numeric_features)):
        for j in range(i+1, len(numeric_features)):
            feature_i = numeric_features[i]
            feature_j = numeric_features[j]
            corr_value = abs(corr_matrix.loc[feature_i, feature_j])
            
            if corr_value > high_corr_threshold:
                high_corr_pairs.append((feature_i, feature_j, corr_value))
    
    if high_corr_pairs:
        print("\nHighly correlated feature pairs (correlation > 0.5):")
        for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True):
            print(f"  • {feat1} & {feat2}: {corr:.4f}")
    else:
        print("\nNo highly correlated feature pairs found (all correlations <= 0.5)")
    
    # Calculate VIF for numeric features
    print("\n2. Calculating Variance Inflation Factors (VIF)...")
    
    # Standardize the numeric features for VIF calculation
    X_numeric = df[numeric_features].copy()
    scaler = StandardScaler()
    X_numeric_scaled = pd.DataFrame(
        scaler.fit_transform(X_numeric),
        columns=numeric_features
    )
    
    vif_results = calculate_vif(X_numeric_scaled)
    print("\nVIF for each feature:")
    print(vif_results.to_string(index=False))
    
    # Identify features with high VIF
    vif_threshold = 5
    high_vif_features = vif_results[vif_results["VIF"] > vif_threshold]
    
    if not high_vif_features.empty:
        print(f"\nFeatures with high multicollinearity (VIF > {vif_threshold}):")
        for _, row in high_vif_features.iterrows():
            print(f"  • {row['Feature']}: {row['VIF']:.2f}")
    else:
        print(f"\nNo features with high multicollinearity (all VIF <= {vif_threshold})")
    
    # Calculate correlations with target
    print("\n3. Analyzing feature-target correlations...")
    target_corr = calculate_feature_correlations_with_target(df, numeric_features, target)
    print("\nFeature correlations with target variable:")
    print(target_corr.to_string(index=False))
    
    # Visualize feature-target correlations
    plot_feature_target_correlations(target_corr)
    
    # Plot feature relationships with target
    print("\n4. Visualizing feature-target relationships...")
    plot_feature_relationships(df, numeric_features, target)
    
    # Analyze categorical features
    print("\n5. Analyzing categorical features...")
    analyze_categorical_features(df, categorical_features, target)
    
    # Provide recommendations based on findings
    print("\n=== SUMMARY AND RECOMMENDATIONS ===")
    
    # Handle high multicollinearity cases
    if high_vif_features.empty and not high_corr_pairs:
        print("\n✓ No significant multicollinearity detected among the features.")
        print("  All features can be safely used in your models.")
    else:
        print("\n⚠️ Multicollinearity detected in some features:")
        
        if not high_vif_features.empty:
            print(f"\n  Features with high VIF (> {vif_threshold}):")
            for _, row in high_vif_features.iterrows():
                print(f"  • {row['Feature']}: {row['VIF']:.2f}")
        
        if high_corr_pairs:
            print("\n  Highly correlated feature pairs:")
            for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True):
                print(f"  • {feat1} & {feat2}: {corr:.4f}")
        
        print("\nPotential solutions:")
        print("  1. Feature selection: Remove one feature from each highly correlated pair")
        print("  2. Feature transformation: Create composite features or use dimensionality reduction")
        print("  3. Regularization: Use L1 or L2 regularization (already in Elastic Net and XGBoost)")
        print("  4. Try tree-based models (Random Forest, XGBoost) which are less affected by multicollinearity")
    
    # Recommend most important features based on correlation
    print("\nMost important features based on correlation with target:")
    top_features = target_corr.head(3)
    for _, row in top_features.iterrows():
        feature = row['Feature']
        pearson = row['Pearson_Correlation']
        spearman = row['Spearman_Correlation']
        print(f"  • {feature}: Pearson={pearson:.4f}, Spearman={spearman:.4f}")
    
    # Overall model strategy recommendations
    print("\nRecommendations for model strategy:")
    
    if 'Treated_astig' in top_features['Feature'].values:
        print("  ✓ The monotonic constraints on 'Treated_astig' in XGBoost and Neural Network models")
        print("    are well-justified based on its correlation with the target variable.")
    
    if high_vif_features.empty and not high_corr_pairs:
        print("  ✓ All current features can be retained in the models.")
    else:
        print("  ⚠️ Consider evaluating models with and without highly correlated features to")
        print("    see if removing them improves performance and interpretability.")
    
    # Regularization recommendations
    if not high_vif_features.empty or high_corr_pairs:
        print("  ✓ The Elastic Net model is a good choice as it includes L1 regularization")
        print("    which helps manage multicollinearity through feature selection.")
    
    print("\nAnalysis complete. Results and visualizations saved to output directory.")

if __name__ == "__main__":
    main() 