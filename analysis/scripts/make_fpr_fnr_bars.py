#!/usr/bin/env python3
"""
FPR/FNR Bar Chart Visualization Script
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = "results"
OUT_DIR = os.path.join("analysis", "data", "Figures")
os.makedirs(OUT_DIR, exist_ok=True)

def plot_bars(df, scenario, metric):
    """Create bar plot for FPR or FNR metrics."""
    df_sorted = df.sort_values(metric, ascending=True)
    
    plt.figure(figsize=(10, 6))
    colors = []
    binary_models = ['webrtc', 'whisper_tiny', 'whisper_small']
    
    for model in df_sorted['model']:
        if any(bin_model in model.lower() for bin_model in binary_models):
            colors.append('#FF6B6B')  # Red for binary models
        else:
            colors.append('#4ECDC4')  # Teal for probabilistic models
    
    bars = plt.bar(df_sorted['model'], df_sorted[metric], color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(f'{metric.upper()} (Lower is Better)', fontsize=12)
    plt.title(f'{metric.upper()} by Model - {scenario.upper()}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars, df_sorted[metric]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUT_DIR, f"{metric}_bars_{scenario}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path

def main():
    """Main function to generate FPR/FNR visualizations."""
    print("FPR/FNR VISUALIZATION GENERATOR")
    print("=" * 50)
    
    for scenario in ["cmf", "cmfv"]:
        csv_path = os.path.join(RESULTS_DIR, f"summary_fpr_fnr_{scenario}.csv")
        
        if not os.path.exists(csv_path):
            print(f"CSV not found: {csv_path}")
            continue
        
        print(f"\nProcessing scenario: {scenario.upper()}")
        
        df = pd.read_csv(csv_path)
        df_clean = (df.sort_values(["model", "op_name"])
                    .drop_duplicates(subset=["model"], keep="last"))
        
        print(f"Models found: {len(df_clean)}")
        
        for metric in ["fpr", "fnr"]:
            plot_bars(df_clean, scenario, metric)
        
        print(f"Best FPR: {df_clean.loc[df_clean['fpr'].idxmin(), 'model']} ({df_clean['fpr'].min():.3f})")
        print(f"Best FNR: {df_clean.loc[df_clean['fnr'].idxmin(), 'model']} ({df_clean['fnr'].min():.3f})")
    
    print(f"\nVisualization completed!")

if __name__ == "__main__":
    main()
