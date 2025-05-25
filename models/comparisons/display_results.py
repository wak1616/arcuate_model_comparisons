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
    print(f"{'Rank':<4} {'Model':<28} {'R² Score':<12} {'RMSE':<8} {'MAE':<8} {'Notes':<15}")
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
        
        # Add special notes
        notes = ""
        if "Monotonic" in model:
            if "Neural Network" in model:
                notes = "🧠 Monotonic"
            else:
                notes = "🌳 Monotonic"
        elif "Elastic" in model:
            notes = "📐 Regularized"
        elif "Random Forest" in model:
            notes = "🌲 Ensemble"
        elif "XGBoost" in model and "Monotonic" not in model:
            notes = "⚡ Gradient"
        
        print(f"{rank_emoji:<4} {model:<28} {r2_str:<12} {rmse_str:<8} {mae_str:<8} {notes:<15}")
    
    print("-" * 85)
    print()
    
    # Model7 specific analysis
    model7_row = df_sorted[df_sorted['Model'] == 'Monotonic Neural Network']
    if not model7_row.empty:
        model7_rank = model7_row.iloc[0]['Rank']
        model7_r2 = model7_row.iloc[0]['R2_mean']
        model7_rmse = model7_row.iloc[0]['RMSE_mean']
        
        print("🎯 MODEL7 (MONOTONIC NEURAL NETWORK) ANALYSIS:")
        print("-" * 50)
        print(f"🏅 Overall Rank: {model7_rank} out of {len(df_sorted)}")
        print(f"📈 R² Score: {model7_r2:.4f} ({model7_r2*100:.1f}% variance explained)")
        print(f"📊 RMSE: {model7_rmse:.2f}")
        
        # Compare with best model
        best_model = df_sorted.iloc[0]
        best_r2 = best_model['R2_mean']
        best_rmse = best_model['RMSE_mean']
        
        r2_gap = ((best_r2 - model7_r2) / best_r2) * 100
        rmse_gap = ((model7_rmse - best_rmse) / best_rmse) * 100
        
        print(f"🥇 vs Best Model ({best_model['Model']}):")
        print(f"   • R² Gap: -{r2_gap:.1f}% ({model7_r2:.4f} vs {best_r2:.4f})")
        print(f"   • RMSE Gap: +{rmse_gap:.1f}% ({model7_rmse:.2f} vs {best_rmse:.2f})")
        
        # Compare with worst model
        worst_model = df_sorted.iloc[-1]
        worst_r2 = worst_model['R2_mean']
        worst_rmse = worst_model['RMSE_mean']
        
        if worst_model['Model'] != 'Monotonic Neural Network':
            r2_advantage = ((model7_r2 - worst_r2) / worst_r2) * 100
            rmse_advantage = ((worst_rmse - model7_rmse) / worst_rmse) * 100
            
            print(f"🔴 vs Worst Model ({worst_model['Model']}):")
            print(f"   • R² Advantage: +{r2_advantage:.1f}% ({model7_r2:.4f} vs {worst_r2:.4f})")
            print(f"   • RMSE Advantage: -{rmse_advantage:.1f}% ({model7_rmse:.2f} vs {worst_rmse:.2f})")
        
        print()
        
        # Performance tier
        if model7_r2 >= 0.90:
            tier = "🏆 EXCELLENT"
            tier_desc = "Outstanding performance"
        elif model7_r2 >= 0.85:
            tier = "🥇 VERY GOOD"
            tier_desc = "Strong performance"
        elif model7_r2 >= 0.80:
            tier = "🥈 GOOD"
            tier_desc = "Solid performance"
        elif model7_r2 >= 0.70:
            tier = "🥉 FAIR"
            tier_desc = "Acceptable performance"
        else:
            tier = "❌ POOR"
            tier_desc = "Needs improvement"
        
        print(f"🏅 Performance Tier: {tier}")
        print(f"📝 Assessment: {tier_desc}")
        print()
        
        # Unique value proposition
        print("✨ MODEL7 UNIQUE VALUE PROPOSITION:")
        print("-" * 40)
        print("✅ Monotonic Constraints: Ensures Treated_astig ↑ → Arcuate_sweep ↑")
        print("✅ Clinical Interpretability: Relationships make medical sense")
        print("✅ Regulatory Friendly: Predictable, explainable behavior")
        print("✅ No Counterintuitive Predictions: Built-in safety constraints")
        print("⚖️  Trade-off: Sacrifices ~3% performance for guaranteed monotonicity")
        print()
    
    # Summary recommendations
    print("🎯 RECOMMENDATIONS:")
    print("-" * 20)
    print("🏆 Maximum Performance: Use XGBoost Selective-Monotonic")
    print("🔒 Maximum Interpretability: Use Model7 (Monotonic Neural Network)")
    print("⚖️  Balanced Approach: Use XGBoost Selective-Monotonic (good performance + some constraints)")
    print("🏥 Clinical/Regulatory: Use Model7 for guaranteed monotonic behavior")
    print()
    print("="*80)

if __name__ == "__main__":
    main() 