#!/usr/bin/env python3
"""
Model Comparison Results Display

This script displays the complete model comparison results including Model7 (Monotonic Neural Network)
in a nicely formatted table with rankings and performance analysis.
"""

import pandas as pd
import os

def main():
    # Load the comparison results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.join(script_dir, 'model_cv_comparison_results.csv')
    
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        return
    
    # Read the results
    df = pd.read_csv(results_file)
    
    # Extract mean values for ranking
    df['R2_mean'] = df['R²'].apply(lambda x: float(x.split(' ± ')[0]))
    df['RMSE_mean'] = df['RMSE'].apply(lambda x: float(x.split(' ± ')[0]))
    df['MAE_mean'] = df['MAE'].apply(lambda x: float(x.split(' ± ')[0]))
    
    # Sort by R² (descending) for ranking
    df_sorted = df.sort_values('R2_mean', ascending=False).reset_index(drop=True)
    df_sorted['Rank'] = range(1, len(df_sorted) + 1)
    
    print("🏆" + "="*80)
    print("🎯 COMPLETE MODEL COMPARISON RESULTS (5-FOLD CROSS-VALIDATION)")
    print("🏆" + "="*80)
    print()
    
    # Display ranking table
    print("📊 PERFORMANCE RANKING (by R² Score):")
    print("-" * 85)
    print(f"{'Rank':<4} {'Model':<28} {'R² Score':<12} {'RMSE':<8} {'MAE':<8} {'Type':<15}")
    print("-" * 85)
    
    for _, row in df_sorted.iterrows():
        rank = row['Rank']
        model = row['Model']
        r2_str = row['R²']
        rmse_str = row['RMSE']
        mae_str = row['MAE']
        
        # Add rank emoji
        if rank == 1:
            rank_emoji = "🥇"
        elif rank == 2:
            rank_emoji = "🥈"
        elif rank == 3:
            rank_emoji = "🥉"
        else:
            rank_emoji = f"{rank}."
        
        # Add model type classification
        model_type = ""
        if "Monotonic" in model:
            if "Neural Network" in model:
                model_type = "🧠 Neural+Mono"
            else:
                model_type = "🌳 XGB+Mono"
        elif "Elastic" in model:
            model_type = "📐 Linear"
        elif "Random Forest" in model:
            model_type = "🌲 Ensemble"
        elif "XGBoost" in model:
            model_type = "⚡ Gradient"
        
        print(f"{rank_emoji:<4} {model:<28} {r2_str:<12} {rmse_str:<8} {mae_str:<8} {model_type:<15}")
    
    print("-" * 85)
    print()
    
    # Performance statistics
    print("📈 PERFORMANCE STATISTICS:")
    print("-" * 30)
    print(f"Best R² Score: {df_sorted.iloc[0]['R2_mean']:.4f} ({df_sorted.iloc[0]['Model']})")
    print(f"Worst R² Score: {df_sorted.iloc[-1]['R2_mean']:.4f} ({df_sorted.iloc[-1]['Model']})")
    print(f"R² Range: {df_sorted.iloc[0]['R2_mean'] - df_sorted.iloc[-1]['R2_mean']:.4f}")
    print(f"Best RMSE: {df_sorted.iloc[0]['RMSE_mean']:.2f} ({df_sorted.iloc[0]['Model']})")
    print(f"Worst RMSE: {df_sorted.iloc[-1]['RMSE_mean']:.2f} ({df_sorted.iloc[-1]['Model']})")
    print()
    
    # Model characteristics analysis
    print("🔍 MODEL CHARACTERISTICS:")
    print("-" * 30)
    
    # Count model types
    monotonic_models = df_sorted[df_sorted['Model'].str.contains('Monotonic', case=False)]
    xgboost_models = df_sorted[df_sorted['Model'].str.contains('XGBoost', case=False)]
    neural_models = df_sorted[df_sorted['Model'].str.contains('Neural', case=False)]
    
    if not monotonic_models.empty:
        avg_r2_mono = monotonic_models['R2_mean'].mean()
        print(f"Monotonic Models: {len(monotonic_models)} models, Avg R²: {avg_r2_mono:.4f}")
    
    if not xgboost_models.empty:
        avg_r2_xgb = xgboost_models['R2_mean'].mean()
        print(f"XGBoost Models: {len(xgboost_models)} models, Avg R²: {avg_r2_xgb:.4f}")
    
    if not neural_models.empty:
        avg_r2_nn = neural_models['R2_mean'].mean()
        print(f"Neural Network Models: {len(neural_models)} models, Avg R²: {avg_r2_nn:.4f}")
    
    print()
    
    # Model7 specific analysis (if present)
    model7_row = df_sorted[df_sorted['Model'] == 'Monotonic Neural Network']
    if not model7_row.empty:
        model7_rank = model7_row.iloc[0]['Rank']
        model7_r2 = model7_row.iloc[0]['R2_mean']
        model7_rmse = model7_row.iloc[0]['RMSE_mean']
        
        print("🎯 MONOTONIC NEURAL NETWORK (MODEL7) ANALYSIS:")
        print("-" * 50)
        print(f"🏅 Rank: {model7_rank} out of {len(df_sorted)} models")
        print(f"📈 R² Score: {model7_r2:.4f} ({model7_r2*100:.1f}% variance explained)")
        print(f"📊 RMSE: {model7_rmse:.2f}")
        
        # Compare with best model
        best_model = df_sorted.iloc[0]
        best_r2 = best_model['R2_mean']
        best_rmse = best_model['RMSE_mean']
        
        r2_gap = ((best_r2 - model7_r2) / best_r2) * 100
        rmse_gap = ((model7_rmse - best_rmse) / best_rmse) * 100
        
        print(f"📊 vs Best Model ({best_model['Model']}):")
        print(f"   • R² Difference: -{r2_gap:.1f}% ({model7_r2:.4f} vs {best_r2:.4f})")
        print(f"   • RMSE Difference: +{rmse_gap:.1f}% ({model7_rmse:.2f} vs {best_rmse:.2f})")
        
        # Performance tier classification
        if model7_r2 >= 0.90:
            tier = "🏆 EXCELLENT (R² ≥ 0.90)"
        elif model7_r2 >= 0.85:
            tier = "🥇 VERY GOOD (R² ≥ 0.85)"
        elif model7_r2 >= 0.80:
            tier = "🥈 GOOD (R² ≥ 0.80)"
        elif model7_r2 >= 0.70:
            tier = "🥉 FAIR (R² ≥ 0.70)"
        else:
            tier = "❌ POOR (R² < 0.70)"
        
        print(f"🏅 Performance Tier: {tier}")
        
        # Model features (factual, not value propositions)
        print()
        print("🔧 MODEL FEATURES:")
        print("-" * 20)
        print("• Enforces monotonic constraints on specified features")
        print("• Neural network architecture with custom monotonic layers")
        print("• Prevents counterintuitive feature-target relationships")
        print("• Designed for interpretability in clinical contexts")
        print()
    
    # Summary statistics
    print("📋 SUMMARY STATISTICS:")
    print("-" * 25)
    print(f"Total Models Compared: {len(df_sorted)}")
    print(f"R² Score Range: {df_sorted['R2_mean'].min():.4f} - {df_sorted['R2_mean'].max():.4f}")
    print(f"RMSE Range: {df_sorted['RMSE_mean'].min():.2f} - {df_sorted['RMSE_mean'].max():.2f}")
    print(f"Models with R² > 0.90: {len(df_sorted[df_sorted['R2_mean'] > 0.90])}")
    print(f"Models with R² > 0.85: {len(df_sorted[df_sorted['R2_mean'] > 0.85])}")
    print()
    print("="*80)

if __name__ == "__main__":
    main() 