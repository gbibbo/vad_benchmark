#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VAD Parameter Count vs Performance Analysis
Generates publication-ready figures comparing model size vs performance
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analyze_vad_results import (
    load_ground_truth, 
    calculate_metrics_for_model,
    get_available_models,
    get_model_style,
    BINARY_MODELS,
    EXCLUDED_FROM_PLOTS,
    ROOT,
    FIGURES_DIR
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model parameter counts (in millions of parameters)
MODEL_PARAMS = {
    'silero': 0.5,          # 2 MB ≈ 500K params ≈ 0.5M
    'webrtc': 0.01,         # 40 kB ≈ 10K params ≈ 0.01M  
    'panns': 81,            # 81 M params
    'epanns': 24,           # 24 M params (provided)
    'ast': 88,              # 88 M params
    'passt': 90,            # 90 M params
    'whisper_tiny': 39,     # 39 M params
    'whisper_small': 74,    # 74 M params
    'pengi': 1000,          # 1 B params = 1000 M
    'qwen2_audio': 7000     # 7 B params = 7000 M
}

print(f"🔧 Configuración Parameter Count Analysis:")
print(f"   📁 Root: {ROOT}")
print(f"   📁 Figures: {FIGURES_DIR}")
print(f"   📊 Modelos con parameter count: {len(MODEL_PARAMS)}")
for model, params in MODEL_PARAMS.items():
    print(f"      {model}: {params}M params")

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def create_parameter_vs_performance_plot(all_metrics_cmf: dict, all_metrics_cmfv: dict):
    """
    Create parameter count vs performance scatter plot for both scenarios.
    
    🔧 PARA CAMBIAR PROPORCIONES:
    - Línea ~70: fig_width, fig_height 
    - Línea ~140: bbox_to_anchor para posición de leyenda
    """
    print(f"\n📊 Generando scatter Parameter Count vs F1 (CMF vs CMFV)...")
    
    # Prepare data for both scenarios
    def prepare_parameter_data(all_metrics, scenario_name):
        param_data = []
        for model, metrics in all_metrics.items():
            if model in EXCLUDED_FROM_PLOTS:
                continue  # Skip excluded models
            if model not in MODEL_PARAMS:
                print(f"⚠️ {scenario_name} - Modelo sin parameter count: {model}")
                continue
            if metrics["n_matches"] > 0:
                f1_max = np.nanmax(metrics["f1"])
                param_data.append({
                    'Model': model,
                    'F1_max': f1_max,
                    'Params_M': MODEL_PARAMS[model]
                })
                print(f"   🔍 {scenario_name} - {model}: F1={f1_max:.3f}, Params={MODEL_PARAMS[model]}M")
        
        param_df = pd.DataFrame(param_data).set_index('Model')
        
        if param_df.empty:
            print(f"⚠️ No hay datos para {scenario_name}")
            return None
        
        excluded_models = [m for m in param_df.index if m in EXCLUDED_FROM_PLOTS]
        included_models = [m for m in param_df.index if m not in EXCLUDED_FROM_PLOTS]
        
        if excluded_models:
            print(f"   🚫 {scenario_name} - Modelos excluidos: {', '.join(excluded_models)}")
        print(f"   📈 {scenario_name} - Modelos incluidos: {', '.join(included_models)}")
        
        return param_df
    
    # Prepare data for both scenarios
    param_cmf = prepare_parameter_data(all_metrics_cmf, "CMF")
    param_cmfv = prepare_parameter_data(all_metrics_cmfv, "CMFV")
    
    if param_cmf is None or param_cmfv is None:
        print("❌ No se pueden generar gráficos sin datos")
        return
    
    # Get common models for consistent styling
    common_models = sorted(set(param_cmf.index) & set(param_cmfv.index))
    if not common_models:
        print("❌ No hay modelos comunes entre escenarios")
        return
    
    print(f"   🎯 Modelos comunes a graficar: {', '.join(common_models)}")
    
    # Debug: show parameter counts for common models
    print(f"   📋 Parameter counts:")
    for model in common_models:
        print(f"      {model}: {MODEL_PARAMS[model]}M params")
    
    # 🔧 CAMBIAR PROPORCIONES AQUÍ - Gráfico scatter parameter count
    fig_width = 8     # ← ANCHO total
    fig_height = 4    # ← ALTO del gráfico
    
    fig, (ax_cmf, ax_cmfv) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    
    # Store legend elements
    legend_elements = []
    
    # Plot both scenarios with consistent colors
    for i, model in enumerate(common_models):
        # Get consistent color for this model
        colors = plt.cm.Set1(np.linspace(0, 1, min(len(common_models), 9)))
        if len(common_models) > 9:
            colors = plt.cm.tab10(np.linspace(0, 1, len(common_models)))
        
        color = colors[i % len(colors)]
        is_binary = model in BINARY_MODELS
        marker_size = 140 if is_binary else 100
        marker_alpha = 0.8 if is_binary else 0.7
        marker = '^' if is_binary else 'o'  # Different markers for binary vs probabilistic
        
        display_name = model.replace('_', ' ').upper()
        if display_name == 'QWEN2 AUDIO':
            display_name = 'QWEN2'
        
        # Add small jitter to avoid overlapping points (especially AST and PaSST)
        jitter_x = 1.0 + (i - len(common_models)/2) * 0.05  # Small multiplicative jitter
        
        # CMF subplot
        if model in param_cmf.index:
            row_cmf = param_cmf.loc[model]
            params_jittered = row_cmf['Params_M'] * jitter_x
            scatter_cmf = ax_cmf.scatter(params_jittered, row_cmf['F1_max'], 
                                       s=marker_size, 
                                       color=color, marker=marker,
                                       edgecolors='black', linewidth=1)
            
            # Add label only once to legend
            if i == 0 or model not in [elem.get_label() for elem in legend_elements]:
                legend_elements.append(plt.Line2D([0], [0], marker=marker, color='w', 
                                                markerfacecolor=color, markersize=10,
                                                markeredgecolor='black', markeredgewidth=1,
                                                label=display_name, linestyle='None'))
        
        # CMFV subplot  
        if model in param_cmfv.index:
            row_cmfv = param_cmfv.loc[model]
            params_jittered = row_cmfv['Params_M'] * jitter_x
            scatter_cmfv = ax_cmfv.scatter(params_jittered, row_cmfv['F1_max'], 
                                         s=marker_size, 
                                         color=color, marker=marker,
                                         edgecolors='black', linewidth=1)
    
    # Configure CMF subplot (left) - with Y-label
    ax_cmf.set_xlabel('Parameter Count (Millions) → Larger', fontsize=11)
    ax_cmf.set_ylabel('Maximum F1-Score → Better', fontsize=11)
    ax_cmf.set_title('Parameter Count vs Performance - CMF', fontsize=12, fontweight='bold')
    ax_cmf.grid(True, alpha=0.3)
    ax_cmf.set_xscale('log')  # Log scale for parameter count
    ax_cmf.set_xlim([0.005, 10000])  # From 0.01M to 10000M parameters
    ax_cmf.set_ylim([0.6, 1])
    
    # Configure CMFV subplot (right) - NO Y-label
    ax_cmfv.set_xlabel('Parameter Count (Millions) → Larger', fontsize=11)
    ax_cmfv.set_title('Parameter Count vs Performance - CMFV', fontsize=12, fontweight='bold')
    ax_cmfv.grid(True, alpha=0.3)
    ax_cmfv.set_xscale('log')  # Log scale for parameter count
    ax_cmfv.set_xlim([0.005, 10000])  # From 0.01M to 10000M parameters
    ax_cmfv.set_ylim([0.6, 1])
    
    # Add shared legend OUTSIDE the subplots
    fig.legend(handles=legend_elements, 
               bbox_to_anchor=(0.98, 0.5),
               loc='center left',
               fontsize=9,
               frameon=True,
               fancybox=True,
               shadow=True)
    
    # Adjust layout to make room for external legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.82)
    
    # Save plot
    output_path = os.path.join(FIGURES_DIR, "parameter_count_vs_performance_comparison")
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Scatter parameter count comparativo guardado en: {output_path}.png/.pdf")
    
    # Show ranking for both scenarios
    print(f"\n🏆 RANKING DE MODELOS POR EFICIENCIA (F1/Params):")
    
    for scenario_name, param_data in [('CMF', param_cmf), ('CMFV', param_cmfv)]:
        print(f"\n📊 {scenario_name}:")
        
        # Calculate efficiency (F1 per million parameters)
        param_data_copy = param_data.copy()
        param_data_copy['Efficiency'] = param_data_copy['F1_max'] / param_data_copy['Params_M']
        
        # Sort by efficiency (best F1 per parameter)
        efficiency_sorted = param_data_copy[~param_data_copy.index.isin(EXCLUDED_FROM_PLOTS)].sort_values('Efficiency', ascending=False)
        
        print("   Por eficiencia (F1/M params):")
        for rank, (model, row) in enumerate(efficiency_sorted.iterrows(), 1):
            display_name = model.replace('_', ' ').upper()
            print(f"   {rank}. {display_name}: F1={row['F1_max']:.3f}, Params={row['Params_M']:.1f}M, Eff={row['Efficiency']:.4f}")
        
        print(f"\n   Por F1 absoluto:")
        f1_sorted = param_data_copy[~param_data_copy.index.isin(EXCLUDED_FROM_PLOTS)].sort_values('F1_max', ascending=False)
        for rank, (model, row) in enumerate(f1_sorted.iterrows(), 1):
            display_name = model.replace('_', ' ').upper()
            print(f"   {rank}. {display_name}: F1={row['F1_max']:.3f}, Params={row['Params_M']:.1f}M")
    
    if any(model in EXCLUDED_FROM_PLOTS for model in (set(param_cmf.index) | set(param_cmfv.index))):
        print(f"\n📋 Modelos excluidos de gráficos: {', '.join(EXCLUDED_FROM_PLOTS)}")


def create_single_scenario_parameter_plot(all_metrics: dict, scenario_name: str):
    """
    Create parameter count vs performance plot for a single scenario.
    
    🔧 PARA CAMBIAR PROPORCIONES DEL GRÁFICO INDIVIDUAL:
    - Línea ~15: figsize=(ancho, alto)
    """
    print(f"\n📊 Generando scatter Parameter Count vs F1 - {scenario_name}...")
    
    # Prepare data
    param_data = []
    for model, metrics in all_metrics.items():
        if model in EXCLUDED_FROM_PLOTS:
            continue  # Skip excluded models
        if model not in MODEL_PARAMS:
            print(f"⚠️ Modelo sin parameter count: {model}")
            continue
        if metrics["n_matches"] > 0:
            f1_max = np.nanmax(metrics["f1"])
            param_data.append({
                'Model': model,
                'F1_max': f1_max,
                'Params_M': MODEL_PARAMS[model]
            })
    
    if not param_data:
        print(f"❌ No hay datos para {scenario_name}")
        return
    
    param_df = pd.DataFrame(param_data)
    
    # 🔧 CAMBIAR PROPORCIONES AQUÍ - Gráfico individual
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot with different styles for binary models
    for i, row in param_df.iterrows():
        model = row['Model']
        is_binary = model in BINARY_MODELS
        
        # Get color and style
        colors = plt.cm.Set1(np.linspace(0, 1, min(len(param_df), 9)))
        if len(param_df) > 9:
            colors = plt.cm.tab10(np.linspace(0, 1, len(param_df)))
        
        color = colors[i % len(colors)]
        marker_size = 140 if is_binary else 100
        marker = '^' if is_binary else 'o'
        
        display_name = model.replace('_', ' ').upper()
        if display_name == 'QWEN2 AUDIO':
            display_name = 'QWEN2'
        
        # Add small jitter to avoid overlapping points
        jitter_x = 1.0 + (i - len(param_df)/2) * 0.05
        params_jittered = row['Params_M'] * jitter_x
        
        scatter = ax.scatter(params_jittered, row['F1_max'], 
                           s=marker_size, color=color, marker=marker,
                           edgecolors='black', linewidth=1,
                           label=display_name)
    
    # Configure plot
    ax.set_xlabel('Parameter Count (Millions) → Larger', fontsize=12)
    ax.set_ylabel('Maximum F1-Score → Better', fontsize=12)
    ax.set_title(f'Parameter Count vs Performance - {scenario_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xlim([0.005, 10000])
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=10)
    
    # Save plot
    output_path = os.path.join(FIGURES_DIR, f"parameter_count_vs_performance_{scenario_name.lower()}")
    fig.tight_layout()
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Scatter {scenario_name} guardado en: {output_path}.png/.pdf")


def save_parameter_analysis_table(all_metrics_cmf: dict, all_metrics_cmfv: dict):
    """Save parameter count analysis table."""
    
    analysis_data = []
    
    for model in MODEL_PARAMS.keys():
        if model in EXCLUDED_FROM_PLOTS:
            continue
            
        row = {
            'Model': model.replace('_', ' ').upper(),
            'Params_M': MODEL_PARAMS[model],
            'Is_Binary': model in BINARY_MODELS
        }
        
        # CMF metrics
        if model in all_metrics_cmf and all_metrics_cmf[model]["n_matches"] > 0:
            row['F1_max_CMF'] = np.nanmax(all_metrics_cmf[model]["f1"])
            row['Efficiency_CMF'] = row['F1_max_CMF'] / row['Params_M']
        else:
            row['F1_max_CMF'] = None
            row['Efficiency_CMF'] = None
        
        # CMFV metrics
        if model in all_metrics_cmfv and all_metrics_cmfv[model]["n_matches"] > 0:
            row['F1_max_CMFV'] = np.nanmax(all_metrics_cmfv[model]["f1"])
            row['Efficiency_CMFV'] = row['F1_max_CMFV'] / row['Params_M']
        else:
            row['F1_max_CMFV'] = None
            row['Efficiency_CMFV'] = None
        
        analysis_data.append(row)
    
    analysis_df = pd.DataFrame(analysis_data)
    
    # Sort by average efficiency
    analysis_df['Avg_Efficiency'] = (analysis_df['Efficiency_CMF'].fillna(0) + 
                                    analysis_df['Efficiency_CMFV'].fillna(0)) / 2
    analysis_df = analysis_df.sort_values('Avg_Efficiency', ascending=False)
    
    # Save table
    output_path = os.path.join(FIGURES_DIR, 'parameter_count_analysis.csv')
    analysis_df.to_csv(output_path, index=False)
    
    print(f"   📊 Tabla de análisis de parámetros guardada en: {output_path}")
    
    return analysis_df


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main parameter count analysis function."""
    print("🚀 ANÁLISIS VAD - PARAMETER COUNT VS PERFORMANCE")
    print("=" * 60)
    
    # 1. Get available models
    models = get_available_models()
    if not models:
        raise RuntimeError("❌ No se encontraron modelos (carpetas masks_*)")
    
    print(f"🤖 Modelos encontrados: {', '.join(models)}")
    if EXCLUDED_FROM_PLOTS:
        print(f"🚫 Modelos excluidos de gráficos: {', '.join(EXCLUDED_FROM_PLOTS)}")
    
    # 2. Analyze both scenarios
    print(f"\n{'='*60}")
    print(f"📋 ANALIZANDO ESCENARIO: CMF")
    print(f"{'='*60}")
    
    gt_cmf = load_ground_truth('cmf')
    all_metrics_cmf = {}
    for model in models:
        if model in MODEL_PARAMS:  # Only analyze models with known parameter count
            all_metrics_cmf[model] = calculate_metrics_for_model(model, gt_cmf)
    
    print(f"\n{'='*60}")
    print(f"📋 ANALIZANDO ESCENARIO: CMFV")
    print(f"{'='*60}")
    
    gt_cmfv = load_ground_truth('cmfv')
    all_metrics_cmfv = {}
    for model in models:
        if model in MODEL_PARAMS:  # Only analyze models with known parameter count
            all_metrics_cmfv[model] = calculate_metrics_for_model(model, gt_cmfv)
    
    # 3. Create parameter count vs performance plots
    print(f"\n{'='*60}")
    print("📊 GENERANDO GRÁFICOS PARAMETER COUNT VS PERFORMANCE")
    print(f"{'='*60}")
    
    # Side-by-side comparison
    create_parameter_vs_performance_plot(all_metrics_cmf, all_metrics_cmfv)
    
    # Individual plots for each scenario
    create_single_scenario_parameter_plot(all_metrics_cmf, "CMF")
    create_single_scenario_parameter_plot(all_metrics_cmfv, "CMFV")
    
    # 4. Save analysis table
    analysis_df = save_parameter_analysis_table(all_metrics_cmf, all_metrics_cmfv)
    
    # 5. Final summary
    print(f"\n{'='*60}")
    print("🎯 RESUMEN PARAMETER COUNT ANALYSIS")
    print(f"{'='*60}")
    
    print(f"\n🏆 TOP 3 MODELOS MÁS EFICIENTES (F1/M params):")
    for rank, (_, row) in enumerate(analysis_df.head(3).iterrows(), 1):
        print(f"   {rank}. {row['Model']}: Avg Eff = {row['Avg_Efficiency']:.4f}")
        print(f"      Params: {row['Params_M']:.1f}M, CMF F1: {row['F1_max_CMF']:.3f}, CMFV F1: {row['F1_max_CMFV']:.3f}")
    
    print(f"\n✅ Análisis de parameter count completado!")
    print(f"📁 Resultados en: {FIGURES_DIR}")

if __name__ == "__main__":
    main()