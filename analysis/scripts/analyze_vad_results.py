#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VAD Performance Analysis - Version with Side-by-Side Comparison
Generates publication-ready figures comparing CMF vs CMFV scenarios
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import norm
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, 
    confusion_matrix
)

# ============================================================================
# CONFIGURATION
# ============================================================================
from pathlib import Path; ROOT = str(Path(__file__).parent.parent / "data")
GT_DIR = f"{ROOT}/ground_truth_chime"
MASK_DIR = f"{ROOT}/results_CHiME"
GT_FILES = {"cmf": "cmf.csv", "cmfv": "cmfv.csv"}
LOG_NAME = "evaluation_20250707_075929.log"

# Thresholds and binary models
THRESHOLDS = [f"{t/100:.2f}" for t in range(0, 101, 5)]
BINARY_MODELS = ["qwen2_audio", "webrtc", "whisper_small", "whisper_tiny"]  # Models that produce binary decisions

# Models to exclude from plots (but keep in analysis)
EXCLUDED_FROM_PLOTS = ["pengi", "qwen2_audio"]

# Create main figures directory
FIGURES_DIR = os.path.join(ROOT, "Figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

print(f"🔧 Configuración:")
print(f"   📁 Root: {ROOT}")
print(f"   📁 Figures: {FIGURES_DIR}")
print(f"   🎯 Thresholds: {len(THRESHOLDS)} ({THRESHOLDS[0]} a {THRESHOLDS[-1]})")
print(f"   🤖 Modelos binarios: {BINARY_MODELS}")

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def load_ground_truth(scenario: str) -> pd.DataFrame:
    """Load and process ground truth data."""
    gt_path = os.path.join(GT_DIR, GT_FILES[scenario])
    
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"❌ Ground truth no encontrado: {gt_path}")
    
    gt = pd.read_csv(gt_path)
    print(f"📋 GT {scenario.upper()}: {len(gt)} archivos cargados")
    
    # Convert format: Chunk -> Filename, Condition -> Speech_gt
    gt_processed = pd.DataFrame({
        'Filename': gt['Chunk'].astype(str) + '.16kHz.wav',
        'Speech_gt': gt['Condition'].astype(int)
    })
    
    speech_count = gt_processed['Speech_gt'].sum()
    no_speech_count = len(gt_processed) - speech_count
    print(f"   📊 Speech=1: {speech_count}, Speech=0: {no_speech_count}")
    
    return gt_processed

def load_prediction_mask(model: str, threshold: str) -> pd.DataFrame:
    """Load prediction mask for a model and threshold."""
    mask_path = os.path.join(MASK_DIR, f"masks_{model}", f"mask_{threshold}.csv")
    
    if not os.path.exists(mask_path):
        return None
    
    mask = pd.read_csv(mask_path)
    mask['Speech_pred'] = mask['Speech'].astype(int)
    
    return mask[['Filename', 'Speech_pred']]

def get_available_models() -> list:
    """Get list of available models."""
    models = []
    for item in os.listdir(MASK_DIR):
        if item.startswith("masks_"):
            model_name = item.replace("masks_", "")
            models.append(model_name)
    return sorted(models)

def calculate_metrics_for_model(model: str, gt: pd.DataFrame) -> dict:
    """Calculate metrics for a model across all thresholds."""
    print(f"📊 Calculando métricas para {model.upper()}...")
    
    n_thresholds = len(THRESHOLDS)
    precision = np.full(n_thresholds, np.nan)
    recall = np.full(n_thresholds, np.nan)
    f1 = np.full(n_thresholds, np.nan)
    accuracy = np.full(n_thresholds, np.nan)
    fpr = np.full(n_thresholds, np.nan)
    
    # Special case: binary models
    if model in BINARY_MODELS:
        print(f"   🎯 Modelo binario detectado: {model}")
        
        mask = load_prediction_mask(model, "0.00")
        if mask is None:
            print(f"   ❌ No se pudo cargar mask para {model}")
            return {"precision": precision, "recall": recall, "f1": f1, 
                   "accuracy": accuracy, "fpr": fpr, "n_matches": 0}
        
        merged = gt.merge(mask, on='Filename', how='inner')
        if merged.empty:
            print(f"   ❌ No hay matches entre GT y predicciones para {model}")
            return {"precision": precision, "recall": recall, "f1": f1, 
                   "accuracy": accuracy, "fpr": fpr, "n_matches": 0}
        
        y_true = merged['Speech_gt'].values
        y_pred = merged['Speech_pred'].values
        
        prec_val = precision_score(y_true, y_pred, zero_division=0)
        rec_val = recall_score(y_true, y_pred, zero_division=0)
        f1_val = f1_score(y_true, y_pred, zero_division=0)
        acc_val = accuracy_score(y_true, y_pred)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Replicate for all thresholds
        precision.fill(prec_val)
        recall.fill(rec_val)
        f1.fill(f1_val)
        accuracy.fill(acc_val)
        fpr.fill(fpr_val)
        
        print(f"   ✅ Métricas binarias: P={prec_val:.3f}, R={rec_val:.3f}, F1={f1_val:.3f}")
        print(f"   🔍 Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"   📊 FPR={fpr_val:.3f}, TPR={rec_val:.3f}")
        print(f"   ⚠️  Checking: Always predicts speech? FN={fn}, TN={tn}")
        print(f"   📝 Replicando valores para todos los {n_thresholds} thresholds")
        
        return {"precision": precision, "recall": recall, "f1": f1, 
               "accuracy": accuracy, "fpr": fpr, "n_matches": len(merged)}
    
    # Normal case: probabilistic models
    matches_count = 0
    
    for i, threshold in enumerate(THRESHOLDS):
        mask = load_prediction_mask(model, threshold)
        if mask is None:
            continue
        
        merged = gt.merge(mask, on='Filename', how='inner')
        if merged.empty:
            continue
            
        matches_count = len(merged)
        
        y_true = merged['Speech_gt'].values
        y_pred = merged['Speech_pred'].values
        
        precision[i] = precision_score(y_true, y_pred, zero_division=0)
        recall[i] = recall_score(y_true, y_pred, zero_division=0)
        f1[i] = f1_score(y_true, y_pred, zero_division=0)
        accuracy[i] = accuracy_score(y_true, y_pred)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fpr[i] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # Report valid metrics
    valid_metrics = ~np.isnan(f1)
    n_valid = valid_metrics.sum()
    
    if n_valid > 0:
        best_f1_idx = np.nanargmax(f1)
        best_f1 = f1[best_f1_idx]
        best_threshold = THRESHOLDS[best_f1_idx]
        print(f"   ✅ {n_valid}/{n_thresholds} thresholds válidos")
        print(f"   🎯 Mejor F1: {best_f1:.3f} @ threshold {best_threshold}")
    else:
        print(f"   ❌ No se pudieron calcular métricas válidas")
    
    return {"precision": precision, "recall": recall, "f1": f1, 
           "accuracy": accuracy, "fpr": fpr, "n_matches": matches_count}

def safe_auc(x, y, is_binary_model=False):
    """Calculate AUC safely, handling NaN values and binary models."""
    if is_binary_model:
        # For binary models, AUC is not mathematically meaningful
        return float('nan')  # Will display as "N/A" in labels
    
    valid_mask = ~np.isnan(x) & ~np.isnan(y)
    if valid_mask.sum() < 2:
        return 0.5
    
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    
    # Check if all values are the same (effectively binary behavior)
    if len(np.unique(x_valid)) <= 1 or len(np.unique(y_valid)) <= 1:
        return float('nan')  # No curve to measure
    
    sort_idx = np.argsort(x_valid)
    x_sorted = x_valid[sort_idx]
    y_sorted = y_valid[sort_idx]
    
    return np.trapz(y_sorted, x_sorted)

def format_auc_label(auc_value):
    """Format AUC value for display, handling NaN."""
    if np.isnan(auc_value):
        return "N/A"
    else:
        return f"{auc_value:.3f}"

def get_model_style(model: str, model_idx: int, total_models: int, plot_type: str = "line"):
    """
    Get color and marker style for a model.
    
    Args:
        plot_type: "line" for continuous lines, "point" for single points (binary models in ROC/PR)
    """
    # Color palette
    colors = plt.cm.Set1(np.linspace(0, 1, min(total_models, 9)))
    if total_models > 9:
        colors = plt.cm.tab10(np.linspace(0, 1, total_models))
    
    # Markers for single points
    markers = ['o', '^', 's', 'v', '<', '>', 'D', 'p', 'h', '*']
    
    if plot_type == "point":
        # For binary models in ROC/PR - single point (for scatter())
        return {
            'color': colors[model_idx % len(colors)],
            'marker': markers[model_idx % len(markers)],
            's': 100,  # Size for scatter (not markersize)
            'edgecolors': 'black',
            'linewidths': 1
        }
    else:
        # For continuous lines - no markers (for plot())
        return {
            'color': colors[model_idx % len(colors)],
            'marker': 'None',  # No markers on lines
            'markersize': 0,
            'linewidth': 2,    # Slightly thicker line
            'linestyle': '-'   # Continuous line
        }

def create_side_by_side_plots(all_metrics_cmf: dict, all_metrics_cmfv: dict):
    """
    Create side-by-side comparison plots for CMF vs CMFV scenarios.
    
    🔧 PARA CAMBIAR PROPORCIONES DE GRÁFICOS LADO A LADO:
    - Línea ~210: fig_width, fig_height 
    - Ejemplo: fig_width=10, fig_height=4 (más pequeño)
    - Ejemplo: fig_width=14, fig_height=6 (más grande)
    """
    print(f"\n📈 Generando gráficos comparativos lado a lado...")
    
    # Get common models and filter out excluded ones
    models_cmf = set(all_metrics_cmf.keys())
    models_cmfv = set(all_metrics_cmfv.keys())
    common_models = sorted((models_cmf & models_cmfv) - set(EXCLUDED_FROM_PLOTS))
    
    if not common_models:
        print("❌ No hay modelos comunes entre escenarios")
        return
    
    print(f"   📊 Modelos incluidos en gráficos: {', '.join(common_models)}")
    print(f"   🚫 Modelos excluidos: {', '.join(EXCLUDED_FROM_PLOTS)}")
    
    # 🔧 CAMBIAR PROPORCIONES AQUÍ - Gráficos lado a lado
    fig_width = 8    # ← ANCHO total (izquierda + derecha)
    fig_height = 4    # ← ALTO de cada gráfico
    
    # 1. ROC Curves Side-by-Side
    fig, (ax_cmf, ax_cmfv) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    
    # Plot models on both subplots
    for i, model in enumerate(common_models):
        display_name = model.replace('_', ' ').upper()
        if display_name == 'QWEN2 AUDIO':
            display_name = 'QWEN2'
        
        # Determine if binary model
        is_binary = model in BINARY_MODELS
        
        # CMF subplot
        metrics_cmf = all_metrics_cmf[model]
        if metrics_cmf["n_matches"] > 0:
            fpr_cmf = metrics_cmf["fpr"]
            recall_cmf = metrics_cmf["recall"]
            roc_auc_cmf = safe_auc(fpr_cmf, recall_cmf)
            
            if is_binary:
                # Binary model: single point - NO AUC label
                style = get_model_style(model, i, len(common_models), "point")
                ax_cmf.scatter(fpr_cmf[0], recall_cmf[0], 
                             label=f'{display_name}',  # ← Solo nombre, sin AUC
                             **style)
            else:
                # Probabilistic model: continuous line
                style = get_model_style(model, i, len(common_models), "line")
                roc_auc_cmf = safe_auc(fpr_cmf, recall_cmf, is_binary_model=False)
                auc_label = format_auc_label(roc_auc_cmf)
                ax_cmf.plot(fpr_cmf, recall_cmf, 
                           label=f'{display_name} (AUC={auc_label})',
                           **style)
        
        # CMFV subplot
        metrics_cmfv = all_metrics_cmfv[model]
        if metrics_cmfv["n_matches"] > 0:
            fpr_cmfv = metrics_cmfv["fpr"]
            recall_cmfv = metrics_cmfv["recall"]
            roc_auc_cmfv = safe_auc(fpr_cmfv, recall_cmfv)
            
            if is_binary:
                # Binary model: single point - NO AUC label
                style = get_model_style(model, i, len(common_models), "point")
                ax_cmfv.scatter(fpr_cmfv[0], recall_cmfv[0],
                               label=f'{display_name}',  # ← Solo nombre, sin AUC
                               **style)
            else:
                # Probabilistic model: continuous line
                style = get_model_style(model, i, len(common_models), "line")
                roc_auc_cmfv = safe_auc(fpr_cmfv, recall_cmfv, is_binary_model=False)
                auc_label = format_auc_label(roc_auc_cmfv)
                ax_cmfv.plot(fpr_cmfv, recall_cmfv,
                            label=f'{display_name} (AUC={auc_label})',
                            **style)
    
    # Configure ROC plots
    # Left plot (CMF) - with Y-label and legend
    ax_cmf.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    ax_cmf.set_xlabel('False Positive Rate')
    ax_cmf.set_ylabel('True Positive Rate')  # ← Solo eje Y izquierdo
    ax_cmf.set_title('ROC Curves - CMF')
    ax_cmf.grid(True, alpha=0.3)
    ax_cmf.set_xlim([0, 1.05])  # ← Ampliar márgenes para ver puntos extremos
    ax_cmf.set_ylim([0, 1.05])  # ← Ampliar márgenes para ver puntos extremos
    ax_cmf.legend(fontsize=9)  # ← Solo leyenda izquierda
    
    # Right plot (CMFV) - NO Y-label, NO legend
    ax_cmfv.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    ax_cmfv.set_xlabel('False Positive Rate')
    # ax_cmfv.set_ylabel()  ← SIN ylabel para ahorrar espacio
    ax_cmfv.set_title('ROC Curves - CMFV')
    ax_cmfv.grid(True, alpha=0.3)
    ax_cmfv.set_xlim([0, 1.05])  # ← Ampliar márgenes para ver puntos extremos
    ax_cmfv.set_ylim([0, 1.05])  # ← Ampliar márgenes para ver puntos extremos
    ax_cmfv.legend(fontsize=9)  #← SIN leyenda para ahorrar espacio
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'roc_curves_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'roc_curves_comparison.pdf'), 
                bbox_inches='tight')
    plt.close()
    
    # 2. Precision-Recall Curves Side-by-Side
    fig, (ax_cmf, ax_cmfv) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    
    for i, model in enumerate(common_models):
        display_name = model.replace('_', ' ').upper()
        if display_name == 'QWEN2 AUDIO':
            display_name = 'QWEN2'
        
        # Determine if binary model
        is_binary = model in BINARY_MODELS
        
        # CMF subplot
        metrics_cmf = all_metrics_cmf[model]
        if metrics_cmf["n_matches"] > 0:
            precision_cmf = metrics_cmf["precision"]
            recall_cmf = metrics_cmf["recall"]
            pr_auc_cmf = safe_auc(recall_cmf, precision_cmf)
            
            if is_binary:
                # Binary model: single point - NO AP label
                style = get_model_style(model, i, len(common_models), "point")
                ax_cmf.scatter(recall_cmf[0], precision_cmf[0],
                              label=f'{display_name}',  # ← Solo nombre, sin AP
                              **style)
            else:
                # Probabilistic model: continuous line
                style = get_model_style(model, i, len(common_models), "line")
                pr_auc_cmf = safe_auc(recall_cmf, precision_cmf, is_binary_model=False)
                auc_label = format_auc_label(pr_auc_cmf)
                ax_cmf.plot(recall_cmf, precision_cmf,
                           label=f'{display_name} (AP={auc_label})',
                           **style)
        
        # CMFV subplot
        metrics_cmfv = all_metrics_cmfv[model]
        if metrics_cmfv["n_matches"] > 0:
            precision_cmfv = metrics_cmfv["precision"]
            recall_cmfv = metrics_cmfv["recall"]
            pr_auc_cmfv = safe_auc(recall_cmfv, precision_cmfv)
            
            if is_binary:
                # Binary model: single point - NO AP label
                style = get_model_style(model, i, len(common_models), "point")
                ax_cmfv.scatter(recall_cmfv[0], precision_cmfv[0],
                               label=f'{display_name}',  # ← Solo nombre, sin AP
                               **style)
            else:
                # Probabilistic model: continuous line
                style = get_model_style(model, i, len(common_models), "line")
                pr_auc_cmfv = safe_auc(recall_cmfv, precision_cmfv, is_binary_model=False)
                auc_label = format_auc_label(pr_auc_cmfv)
                ax_cmfv.plot(recall_cmfv, precision_cmfv,
                            label=f'{display_name} (AP={auc_label})',
                            **style)
    
    # Configure PR plots
    # Left plot (CMF) - with Y-label and legend
    ax_cmf.set_xlabel('Recall')
    ax_cmf.set_ylabel('Precision')  # ← Solo eje Y izquierdo
    ax_cmf.set_title('Precision-Recall Curves - CMF')
    ax_cmf.grid(True, alpha=0.3)
    ax_cmf.set_xlim([0, 1.05])  # ← Ampliar márgenes para ver puntos extremos
    ax_cmf.set_ylim([0, 1.05])  # ← Ampliar márgenes para ver puntos extremos
    ax_cmf.legend(fontsize=9)  # ← Solo leyenda izquierda
    
    # Right plot (CMFV) - NO Y-label, NO legend
    ax_cmfv.set_xlabel('Recall')
    # ax_cmfv.set_ylabel()  ← SIN ylabel para ahorrar espacio
    ax_cmfv.set_title('Precision-Recall Curves - CMFV')
    ax_cmfv.grid(True, alpha=0.3)
    ax_cmfv.set_xlim([0, 1.05])  # ← Ampliar márgenes para ver puntos extremos
    ax_cmfv.set_ylim([0, 1.05])  # ← Ampliar márgenes para ver puntos extremos
    # ax_cmfv.legend()  ← SIN leyenda para ahorrar espacio
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'pr_curves_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'pr_curves_comparison.pdf'), 
                bbox_inches='tight')
    plt.close()
    
    # 3. F1-Score vs Threshold Side-by-Side
    fig, (ax_cmf, ax_cmfv) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    
    threshold_nums = [float(t) for t in THRESHOLDS]
    
    for i, model in enumerate(common_models):
        # All models use line style (no points)
        style = get_model_style(model, i, len(common_models), "line")
        display_name = model.replace('_', ' ').upper()
        if display_name == 'QWEN2 AUDIO':
            display_name = 'QWEN2'
        
        # CMF subplot
        metrics_cmf = all_metrics_cmf[model]
        if metrics_cmf["n_matches"] > 0:
            f1_cmf = metrics_cmf["f1"]
            ax_cmf.plot(threshold_nums, f1_cmf, label=display_name, **style)
        
        # CMFV subplot
        metrics_cmfv = all_metrics_cmfv[model]
        if metrics_cmfv["n_matches"] > 0:
            f1_cmfv = metrics_cmfv["f1"]
            ax_cmfv.plot(threshold_nums, f1_cmfv, label=display_name, **style)
    
    # Configure F1 plots
    # Left plot (CMF) - with Y-label and legend
    ax_cmf.set_xlabel('Detection Threshold')
    ax_cmf.set_ylabel('F1-Score')  # ← Solo eje Y izquierdo
    ax_cmf.set_title('F1-Score vs Threshold - CMF')
    ax_cmf.grid(True, alpha=0.3)
    ax_cmf.set_xlim([0, 1])
    ax_cmf.set_ylim([0, 1.05])  # ← Ampliar margen superior para ver líneas en 1.0
    ax_cmf.legend(fontsize=9)  # ← Solo leyenda izquierda
    
    # Right plot (CMFV) - NO Y-label, NO legend
    ax_cmfv.set_xlabel('Detection Threshold')
    # ax_cmfv.set_ylabel()  ← SIN ylabel para ahorrar espacio
    ax_cmfv.set_title('F1-Score vs Threshold - CMFV')
    ax_cmfv.grid(True, alpha=0.3)
    ax_cmfv.set_xlim([0, 1])
    ax_cmfv.set_ylim([0, 1.05])  # ← Ampliar margen superior para ver líneas en 1.0
    # ax_cmfv.legend()  ← SIN leyenda para ahorrar espacio
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'f1_vs_threshold_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'f1_vs_threshold_comparison.pdf'), 
                bbox_inches='tight')
    plt.close()
    
    # 4. Accuracy vs Threshold Side-by-Side
    fig, (ax_cmf, ax_cmfv) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    
    for i, model in enumerate(common_models):
        # All models use line style (no points)
        style = get_model_style(model, i, len(common_models), "line")
        display_name = model.replace('_', ' ').upper()
        if display_name == 'QWEN2 AUDIO':
            display_name = 'QWEN2'
        
        # CMF subplot
        metrics_cmf = all_metrics_cmf[model]
        if metrics_cmf["n_matches"] > 0:
            accuracy_cmf = metrics_cmf["accuracy"]
            ax_cmf.plot(threshold_nums, accuracy_cmf, label=display_name, **style)
        
        # CMFV subplot
        metrics_cmfv = all_metrics_cmfv[model]
        if metrics_cmfv["n_matches"] > 0:
            accuracy_cmfv = metrics_cmfv["accuracy"]
            ax_cmfv.plot(threshold_nums, accuracy_cmfv, label=display_name, **style)
    
    # Configure Accuracy plots
    # Left plot (CMF) - with Y-label and legend
    ax_cmf.set_xlabel('Detection Threshold')
    ax_cmf.set_ylabel('Accuracy')  # ← Solo eje Y izquierdo
    ax_cmf.set_title('Accuracy vs Threshold - CMF')
    ax_cmf.grid(True, alpha=0.3)
    ax_cmf.set_xlim([0, 1])
    ax_cmf.set_ylim([0, 1.05])  # ← Ampliar margen superior para ver líneas en 1.0
    ax_cmf.legend(fontsize=9)  # ← Solo leyenda izquierda
    
    # Right plot (CMFV) - NO Y-label, NO legend
    ax_cmfv.set_xlabel('Detection Threshold')
    # ax_cmfv.set_ylabel()  ← SIN ylabel para ahorrar espacio
    ax_cmfv.set_title('Accuracy vs Threshold - CMFV')
    ax_cmfv.grid(True, alpha=0.3)
    ax_cmfv.set_xlim([0, 1])
    ax_cmfv.set_ylim([0, 1.05])  # ← Ampliar margen superior para ver líneas en 1.0
    # ax_cmfv.legend()  ← SIN leyenda para ahorrar espacio
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'accuracy_vs_threshold_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'accuracy_vs_threshold_comparison.pdf'), 
                bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Gráficos comparativos guardados en: {FIGURES_DIR}")

def extract_rtf_from_log(log_path: str, chunk_duration_sec: float = 1.0) -> pd.Series:
    """Extract Real Time Factor from evaluation log."""
    if not os.path.exists(log_path):
        print(f"⚠️ Log no encontrado: {log_path}")
        return pd.Series(dtype=float, name='RTF')
    
    print(f"⏱️ Extrayendo RTF de: {log_path}")
    
    pattern_model = re.compile(r"PROCESANDO MODELO:\s+(\w+)", re.IGNORECASE)
    pattern_file = re.compile(r"📄")
    pattern_time = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
    
    rtf_data = {}
    current_model = None
    processing_times = []
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            model_match = pattern_model.search(line)
            if model_match:
                if current_model and len(processing_times) > 1:
                    time_diffs = np.diff(processing_times)[:100]
                    if len(time_diffs) > 0:
                        avg_processing_time = np.mean(time_diffs.astype('timedelta64[ms]').astype(float)) / 1000
                        rtf = avg_processing_time / chunk_duration_sec
                        rtf_data[current_model] = round(rtf, 4)
                        print(f"   🤖 {current_model}: RTF = {rtf:.4f}")
                
                current_model = model_match.group(1).lower()
                processing_times = []
                continue
            
            if pattern_file.search(line):
                time_match = pattern_time.match(line)
                if time_match:
                    timestamp = datetime.strptime(time_match.group(1), "%Y-%m-%d %H:%M:%S,%f")
                    processing_times.append(timestamp)
    
    # Process last model
    if current_model and len(processing_times) > 1:
        time_diffs = np.diff(processing_times)[:100]
        if len(time_diffs) > 0:
            avg_processing_time = np.mean(time_diffs.astype('timedelta64[ms]').astype(float)) / 1000
            rtf = avg_processing_time / chunk_duration_sec
            rtf_data[current_model] = round(rtf, 4)
            print(f"   🤖 {current_model}: RTF = {rtf:.4f}")
    
    rtf_series = pd.Series(rtf_data, name='RTF')
    rtf_path = os.path.join(FIGURES_DIR, "rtf_by_model.csv")
    rtf_series.to_csv(rtf_path)
    print(f"   💾 RTF guardado en: {rtf_path}")
    
    return rtf_series

def create_performance_vs_speed_plot(all_metrics_cmf: dict, rtf_data: pd.Series):
    """
    Create performance vs speed scatter plot.
    
    PARA CAMBIAR TAMAÑO DEL SCATTER:
    - Línea 395: figsize=(ancho, alto)
    - Ejemplo: figsize=(8, 5) para figura más pequeña
    """
    # Extract F1_max for each model from CMF scenario (excluding filtered models)
    performance_data = []
    
    for model, metrics in all_metrics_cmf.items():
        if model in EXCLUDED_FROM_PLOTS:
            continue  # Skip excluded models
        if metrics["n_matches"] > 0:
            f1_max = np.nanmax(metrics["f1"])
            performance_data.append({
                'Model': model,
                'F1_max': f1_max
            })
    
    performance_df = pd.DataFrame(performance_data).set_index('Model')
    
    # Merge with RTF data
    combined = performance_df.join(rtf_data, how='inner')
    
    if combined.empty:
        print("⚠️ No hay datos combinados para el scatter F1 vs RTF")
        return
    
    print(f"\n📊 Generando scatter F1 vs RTF...")
    excluded_models = [m for m in combined.index if m in EXCLUDED_FROM_PLOTS]
    included_models = [m for m in combined.index if m not in EXCLUDED_FROM_PLOTS]
    
    if excluded_models:
        print(f"   🚫 Modelos excluidos del gráfico: {', '.join(excluded_models)}")
    print(f"   📈 Modelos incluidos en gráfico: {', '.join(included_models)}")
    
    # 🔧 CAMBIAR PROPORCIONES AQUÍ - Gráfico scatter RTF
    fig, ax = plt.subplots(figsize=(6, 4))  # ← (ANCHO, ALTO) del scatter
    
    # Scatter plot with different sizes for binary models
    for model, row in combined.iterrows():
        is_binary = model in BINARY_MODELS
        marker_size = 120 if is_binary else 80
        marker_alpha = 0.8 if is_binary else 0.7
        
        scatter = ax.scatter(row['RTF'], row['F1_max'], 
                           s=marker_size, alpha=marker_alpha, 
                           edgecolors='black', linewidth=1)
    
    # Add model labels
    for model, row in combined.iterrows():
        display_name = model.replace('_', ' ').upper()
        if display_name == 'QWEN2 AUDIO':
            display_name = 'QWEN2'
        
        ax.annotate(display_name, 
                   (row['RTF'], row['F1_max']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Configure axes
    ax.set_xlabel('RTF (Real Time Factor) ← Faster', fontsize=12)
    ax.set_ylabel('Maximum F1-Score (CMF Scenario) → Better', fontsize=12)
    ax.set_title('Performance vs. Computational Speed', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add reference line
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2,
               label='Real-time threshold (RTF = 1.0)')
    ax.legend(fontsize=11)
    
    # Save plot
    output_path = os.path.join(FIGURES_DIR, "performance_vs_speed")
    fig.tight_layout()
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Scatter guardado en: {output_path}.png/.pdf")
    
    # Show ranking (excluding filtered models)
    print(f"\n🏆 RANKING DE MODELOS (en gráficos):")
    combined_sorted = combined[~combined.index.isin(EXCLUDED_FROM_PLOTS)].sort_values('F1_max', ascending=False)
    for rank, (model, row) in enumerate(combined_sorted.iterrows(), 1):
        display_name = model.replace('_', ' ').upper()
        print(f"   {rank}. {display_name}: F1={row['F1_max']:.3f}, RTF={row['RTF']:.3f}")
    
    if any(model in EXCLUDED_FROM_PLOTS for model in combined.index):
        print(f"\n📋 Modelos excluidos de gráficos (pero incluidos en análisis): {', '.join(EXCLUDED_FROM_PLOTS)}")


def create_performance_vs_speed_comparison(all_metrics_cmf: dict, all_metrics_cmfv: dict, rtf_data: pd.Series):
    """
    Create side-by-side performance vs speed scatter plots for CMF vs CMFV.
    
    🔧 PARA CAMBIAR PROPORCIONES:
    - Línea ~20: fig_width, fig_height 
    - Línea ~80: bbox_to_anchor para posición de leyenda
    """
    print(f"\n📊 Generando scatter F1 vs RTF comparativo (CMF vs CMFV)...")
    
    # Prepare data for both scenarios
    def prepare_performance_data(all_metrics, scenario_name):
        performance_data = []
        for model, metrics in all_metrics.items():
            if model in EXCLUDED_FROM_PLOTS:
                continue  # Skip excluded models
            if metrics["n_matches"] > 0:
                f1_max = np.nanmax(metrics["f1"])
                performance_data.append({
                    'Model': model,
                    'F1_max': f1_max
                })
        
        performance_df = pd.DataFrame(performance_data).set_index('Model')
        combined = performance_df.join(rtf_data, how='inner')
        
        if combined.empty:
            print(f"⚠️ No hay datos combinados para {scenario_name}")
            return None
        
        excluded_models = [m for m in combined.index if m in EXCLUDED_FROM_PLOTS]
        included_models = [m for m in combined.index if m not in EXCLUDED_FROM_PLOTS]
        
        if excluded_models:
            print(f"   🚫 {scenario_name} - Modelos excluidos: {', '.join(excluded_models)}")
        print(f"   📈 {scenario_name} - Modelos incluidos: {', '.join(included_models)}")
        
        return combined
    
    # Prepare data for both scenarios
    combined_cmf = prepare_performance_data(all_metrics_cmf, "CMF")
    combined_cmfv = prepare_performance_data(all_metrics_cmfv, "CMFV")
    
    if combined_cmf is None or combined_cmfv is None:
        print("❌ No se pueden generar gráficos comparativos sin datos")
        return
    
    # Get common models for consistent styling
    common_models = sorted(set(combined_cmf.index) & set(combined_cmfv.index))
    if not common_models:
        print("❌ No hay modelos comunes entre escenarios")
        return
    
    # 🔧 CAMBIAR PROPORCIONES AQUÍ - Gráfico scatter comparativo  
    fig_width = 8     # ← ANCHO total optimizado (era 10)
    fig_height = 4    # ← ALTO del gráfico
    
    fig, (ax_cmf, ax_cmfv) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    
    # Store legend elements (to avoid duplicates)
    legend_elements = []
    
    # Plot both scenarios with consistent colors
    for i, model in enumerate(common_models):
        # Get consistent color for this model
        colors = plt.cm.Set1(np.linspace(0, 1, min(len(common_models), 9)))
        if len(common_models) > 9:
            colors = plt.cm.tab10(np.linspace(0, 1, len(common_models)))
        
        color = colors[i % len(colors)]
        is_binary = model in BINARY_MODELS
        marker_size = 140 if is_binary else 100  # ← Marcadores más grandes sin anotaciones
        marker_alpha = 0.8 if is_binary else 0.7
        marker = '^' if is_binary else 'o'  # Different markers for binary vs probabilistic
        
        display_name = model.replace('_', ' ').upper()
        if display_name == 'QWEN2 AUDIO':
            display_name = 'QWEN2'
        
        # CMF subplot
        if model in combined_cmf.index:
            row_cmf = combined_cmf.loc[model]
            scatter_cmf = ax_cmf.scatter(row_cmf['RTF'], row_cmf['F1_max'], 
                                       s=marker_size, #alpha=marker_alpha, 
                                       color=color, marker=marker,
                                       edgecolors='black', linewidth=1)
            
            # Add label only if not already in legend
            if i == 0 or model not in [elem.get_label() for elem in legend_elements]:
                legend_elements.append(plt.Line2D([0], [0], marker=marker, color='w', 
                                                markerfacecolor=color, markersize=10,  # ← Marcador más grande en leyenda
                                                markeredgecolor='black', markeredgewidth=1,
                                                label=display_name, linestyle='None'))
        
        # CMFV subplot  
        if model in combined_cmfv.index:
            row_cmfv = combined_cmfv.loc[model]
            scatter_cmfv = ax_cmfv.scatter(row_cmfv['RTF'], row_cmfv['F1_max'], 
                                         s=marker_size, #alpha=marker_alpha, 
                                         color=color, marker=marker,
                                         edgecolors='black', linewidth=1)
    
    # NO agregar anotaciones de texto sobre los puntos - solo leyenda
    # (Las anotaciones están eliminadas para optimizar espacio visual)
    
    # Configure CMF subplot (left) - with Y-label, NO legend
    ax_cmf.set_xlabel('RTF (Real Time Factor) ← Faster', fontsize=11)
    ax_cmf.set_ylabel('Maximum F1-Score → Better', fontsize=11)  # ← Solo eje Y izquierdo
    ax_cmf.set_title('Performance vs Speed - CMF', fontsize=12, fontweight='bold')
    ax_cmf.grid(True, alpha=0.3)
    ax_cmf.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Configure CMFV subplot (right) - NO Y-label, NO legend
    ax_cmfv.set_xlabel('RTF (Real Time Factor) ← Faster', fontsize=11)
    # ax_cmfv.set_ylabel()  ← SIN ylabel para ahorrar espacio
    ax_cmfv.set_title('Performance vs Speed - CMFV', fontsize=12, fontweight='bold')
    ax_cmfv.grid(True, alpha=0.3)
    ax_cmfv.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Add shared legend OUTSIDE the subplots for maximum space efficiency
    # Add real-time threshold to legend
    legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2,
                                    label='Real-time threshold (RTF = 1.0)'))
    
    # 🔧 CAMBIAR POSICIÓN DE LEYENDA AQUÍ - más cerca para optimizar espacio
    fig.legend(handles=legend_elements, 
               bbox_to_anchor=(0.98, 0.5),   # ← (x, y) - más cerca del plot (era 1.02)
               loc='center left',             # ← ancla de la leyenda
               fontsize=9,                    # ← Fuente más pequeña para ahorrar espacio
               frameon=True,
               fancybox=True,
               shadow=True)
    
    # Adjust layout to make room for external legend - menos espacio reservado
    plt.tight_layout()
    plt.subplots_adjust(right=0.82)  # ← Menos espacio para leyenda (era 0.75)
    
    # Save plot
    output_path = os.path.join(FIGURES_DIR, "performance_vs_speed_comparison")
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Scatter comparativo guardado en: {output_path}.png/.pdf")
    
    # Show ranking for both scenarios
    print(f"\n🏆 RANKING DE MODELOS EN GRÁFICOS:")
    
    for scenario_name, combined in [('CMF', combined_cmf), ('CMFV', combined_cmfv)]:
        print(f"\n📊 {scenario_name}:")
        combined_sorted = combined[~combined.index.isin(EXCLUDED_FROM_PLOTS)].sort_values('F1_max', ascending=False)
        for rank, (model, row) in enumerate(combined_sorted.iterrows(), 1):
            display_name = model.replace('_', ' ').upper()
            print(f"   {rank}. {display_name}: F1={row['F1_max']:.3f}, RTF={row['RTF']:.3f}")
    
    if any(model in EXCLUDED_FROM_PLOTS for model in (set(combined_cmf.index) | set(combined_cmfv.index))):
        print(f"\n📋 Modelos excluidos de gráficos: {', '.join(EXCLUDED_FROM_PLOTS)}")

def save_summary_tables(all_metrics_cmf: dict, all_metrics_cmfv: dict):
    """Save summary tables with performance metrics."""
    
    # CMF Summary
    cmf_summary = []
    for model, metrics in all_metrics_cmf.items():
        if metrics["n_matches"] > 0:
            is_binary = model in BINARY_MODELS
            roc_auc = safe_auc(metrics["fpr"], metrics["recall"], is_binary_model=is_binary)
            pr_auc = safe_auc(metrics["recall"], metrics["precision"], is_binary_model=is_binary)
            
            cmf_summary.append({
                'Model': model.replace('_', ' ').upper(),
                'F1_max': np.nanmax(metrics["f1"]),
                'Precision_max': np.nanmax(metrics["precision"]),
                'Recall_max': np.nanmax(metrics["recall"]),
                'Accuracy_max': np.nanmax(metrics["accuracy"]),
                'ROC_AUC': roc_auc if not np.isnan(roc_auc) else None,
                'PR_AUC': pr_auc if not np.isnan(pr_auc) else None
            })
    
    cmf_df = pd.DataFrame(cmf_summary)
    cmf_df.to_csv(os.path.join(FIGURES_DIR, 'performance_summary_cmf.csv'), index=False)
    
    # CMFV Summary
    cmfv_summary = []
    for model, metrics in all_metrics_cmfv.items():
        if metrics["n_matches"] > 0:
            is_binary = model in BINARY_MODELS
            roc_auc = safe_auc(metrics["fpr"], metrics["recall"], is_binary_model=is_binary)
            pr_auc = safe_auc(metrics["recall"], metrics["precision"], is_binary_model=is_binary)
            
            cmfv_summary.append({
                'Model': model.replace('_', ' ').upper(),
                'F1_max': np.nanmax(metrics["f1"]),
                'Precision_max': np.nanmax(metrics["precision"]),
                'Recall_max': np.nanmax(metrics["recall"]),
                'Accuracy_max': np.nanmax(metrics["accuracy"]),
                'ROC_AUC': roc_auc if not np.isnan(roc_auc) else None,
                'PR_AUC': pr_auc if not np.isnan(pr_auc) else None
            })
    
    cmfv_df = pd.DataFrame(cmfv_summary)
    cmfv_df.to_csv(os.path.join(FIGURES_DIR, 'performance_summary_cmfv.csv'), index=False)
    
    print(f"   📊 Tablas de resumen guardadas en: {FIGURES_DIR}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main analysis function."""
    print("🚀 ANÁLISIS VAD - VERSIÓN CON COMPARACIÓN LADO A LADO")
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
        all_metrics_cmf[model] = calculate_metrics_for_model(model, gt_cmf)
    
    print(f"\n{'='*60}")
    print(f"📋 ANALIZANDO ESCENARIO: CMFV")
    print(f"{'='*60}")
    
    gt_cmfv = load_ground_truth('cmfv')
    all_metrics_cmfv = {}
    for model in models:
        all_metrics_cmfv[model] = calculate_metrics_for_model(model, gt_cmfv)
    
    # 3. Create side-by-side comparison plots
    create_side_by_side_plots(all_metrics_cmf, all_metrics_cmfv)
    
    # 4. RTF Analysis
    print(f"\n{'='*60}")
    print("⏱️ ANÁLISIS DE VELOCIDAD (RTF)")
    print(f"{'='*60}")
    
    log_path = os.path.join(MASK_DIR, LOG_NAME)
    rtf_data = extract_rtf_from_log(log_path)
    
    # 5. Performance vs Speed plot
    if not rtf_data.empty:
        create_performance_vs_speed_plot(all_metrics_cmf, rtf_data)
    
    # 5b. NUEVO: Performance vs Speed comparison (CMF vs CMFV lado a lado)
    if not rtf_data.empty:
        create_performance_vs_speed_comparison(all_metrics_cmf, all_metrics_cmfv, rtf_data)
    
    # 6. Save summary tables
    save_summary_tables(all_metrics_cmf, all_metrics_cmfv)
    
    # 7. Final summary
    print(f"\n{'='*60}")
    print("🎯 RESUMEN FINAL")
    print(f"{'='*60}")
    
    for scenario, all_metrics in [('CMF', all_metrics_cmf), ('CMFV', all_metrics_cmfv)]:
        print(f"\n📊 MEJORES MODELOS - {scenario}:")
        model_scores = []
        for model, metrics in all_metrics.items():
            if metrics["n_matches"] > 0:
                f1_max = np.nanmax(metrics["f1"])
                model_scores.append((model, f1_max))
        
        model_scores.sort(key=lambda x: x[1], reverse=True)
        for rank, (model, f1) in enumerate(model_scores[:3], 1):
            display_name = model.replace('_', ' ').upper()
            print(f"   {rank}. {display_name}: F1={f1:.3f}")
    
    print(f"\n✅ Análisis completado exitosamente!")
    print(f"📁 Resultados en: {FIGURES_DIR}")

if __name__ == "__main__":
    main()