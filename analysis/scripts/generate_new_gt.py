#!/usr/bin/env python3
"""
Comparador de Ground Truth Viejo vs Nuevo
Analiza consistencia y diferencias entre los GT originales y los regenerados
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
from pathlib import Path; RESULTS_ROOT = str(Path(__file__).parent.parent / "data")
OLD_GT_DIR = os.path.join(RESULTS_ROOT, "ground_truth_chime")
NEW_GT_DIR = os.path.join(RESULTS_ROOT, "ground_truth_chime NEW")
COMPARISON_OUTPUT = os.path.join(RESULTS_ROOT, "GT_Comparison_Analysis")

print("🔍 COMPARADOR DE GROUND TRUTH")
print("=" * 50)
print(f"📁 GT Viejo: {OLD_GT_DIR}")
print(f"📁 GT Nuevo: {NEW_GT_DIR}")
print(f"📁 Output: {COMPARISON_OUTPUT}")

# ============================================================================
# FUNCIONES DE CARGA
# ============================================================================

def load_gt_file(gt_dir: str, scenario: str) -> pd.DataFrame:
    """Cargar archivo de ground truth."""
    gt_file = os.path.join(gt_dir, f"{scenario}.csv")
    
    if not os.path.exists(gt_file):
        print(f"⚠️ Archivo GT no encontrado: {gt_file}")
        return pd.DataFrame()
    
    gt = pd.read_csv(gt_file)
    
    # Normalizar formato
    if 'Condition' in gt.columns:
        gt['Speech'] = gt['Condition'].astype(bool).astype(int)
    elif 'Speech' in gt.columns:
        gt['Speech'] = gt['Speech'].astype(bool).astype(int)
    else:
        print(f"❌ Formato GT no reconocido en {gt_file}")
        return pd.DataFrame()
    
    # Asegurar que tenemos la columna Chunk
    if 'Chunk' not in gt.columns and 'Filename' in gt.columns:
        gt['Chunk'] = gt['Filename'].str.replace('.16kHz.wav', '', regex=False)
    
    return gt[['Chunk', 'Speech']].copy()

def load_gt_with_votes(gt_dir: str, scenario: str) -> pd.DataFrame:
    """Cargar archivo de GT con votes (solo para GT nuevo)."""
    gt_file = os.path.join(gt_dir, f"{scenario}_with_votes.csv")
    
    if not os.path.exists(gt_file):
        return pd.DataFrame()
    
    return pd.read_csv(gt_file)

# ============================================================================
# FUNCIONES DE ANÁLISIS
# ============================================================================

def compare_gt_basic_stats(old_gt: pd.DataFrame, new_gt: pd.DataFrame, scenario: str):
    """Comparar estadísticas básicas entre GT viejo y nuevo."""
    print(f"\n📊 ESTADÍSTICAS BÁSICAS - {scenario.upper()}")
    print("-" * 40)
    
    if old_gt.empty:
        print("❌ GT viejo vacío")
        return
    
    if new_gt.empty:
        print("❌ GT nuevo vacío")
        return
    
    # Estadísticas básicas
    old_total = len(old_gt)
    old_speech = old_gt['Speech'].sum()
    old_no_speech = old_total - old_speech
    
    new_total = len(new_gt)
    new_speech = new_gt['Speech'].sum()
    new_no_speech = new_total - new_speech
    
    print(f"GT Viejo:")
    print(f"  Total archivos: {old_total}")
    print(f"  Speech=1: {old_speech} ({old_speech/old_total*100:.1f}%)")
    print(f"  Speech=0: {old_no_speech} ({old_no_speech/old_total*100:.1f}%)")
    
    print(f"\nGT Nuevo:")
    print(f"  Total archivos: {new_total}")
    print(f"  Speech=1: {new_speech} ({new_speech/new_total*100:.1f}%)")
    print(f"  Speech=0: {new_no_speech} ({new_no_speech/new_total*100:.1f}%)")
    
    # Diferencias
    print(f"\nDiferencias:")
    print(f"  Δ Total: {new_total - old_total}")
    print(f"  Δ Speech: {new_speech - old_speech}")
    print(f"  Δ No-Speech: {new_no_speech - old_no_speech}")

def analyze_file_overlap(old_gt: pd.DataFrame, new_gt: pd.DataFrame, scenario: str):
    """Analizar overlap de archivos entre GT viejo y nuevo."""
    print(f"\n🔗 ANÁLISIS DE OVERLAP - {scenario.upper()}")
    print("-" * 40)
    
    if old_gt.empty or new_gt.empty:
        print("❌ No se puede analizar overlap (algún GT está vacío)")
        return None, None, None
    
    old_files = set(old_gt['Chunk'])
    new_files = set(new_gt['Chunk'])
    
    common_files = old_files & new_files
    only_old = old_files - new_files
    only_new = new_files - old_files
    
    print(f"Archivos en común: {len(common_files)}")
    print(f"Solo en GT viejo: {len(only_old)}")
    print(f"Solo en GT nuevo: {len(only_new)}")
    
    if only_old:
        print(f"\nPrimeros 10 solo en GT viejo:")
        for f in list(only_old)[:10]:
            print(f"  - {f}")
    
    if only_new:
        print(f"\nPrimeros 10 solo en GT nuevo:")
        for f in list(only_new)[:10]:
            print(f"  - {f}")
    
    return common_files, only_old, only_new

def analyze_label_changes(old_gt: pd.DataFrame, new_gt: pd.DataFrame, 
                         common_files: set, scenario: str, new_gt_with_votes: pd.DataFrame = None):
    """Analizar cambios de etiquetas en archivos comunes."""
    print(f"\n🔄 ANÁLISIS DE CAMBIOS DE ETIQUETAS - {scenario.upper()}")
    print("-" * 50)
    
    if not common_files or old_gt.empty or new_gt.empty:
        print("❌ No hay archivos comunes para analizar")
        return
    
    # Filtrar a archivos comunes
    old_common = old_gt[old_gt['Chunk'].isin(common_files)].set_index('Chunk').sort_index()
    new_common = new_gt[new_gt['Chunk'].isin(common_files)].set_index('Chunk').sort_index()
    
    # Asegurar mismo orden
    common_sorted = sorted(common_files)
    old_labels = old_common.loc[common_sorted, 'Speech'].values
    new_labels = new_common.loc[common_sorted, 'Speech'].values
    
    # Identificar cambios
    changes = old_labels != new_labels
    num_changes = changes.sum()
    
    print(f"Archivos comunes analizados: {len(common_sorted)}")
    print(f"Archivos con cambio de etiqueta: {num_changes} ({num_changes/len(common_sorted)*100:.1f}%)")
    
    if num_changes == 0:
        print("✅ No hay cambios de etiquetas - GT son idénticos en archivos comunes")
        return
    
    # Tipos de cambios
    old_to_new_0_to_1 = ((old_labels == 0) & (new_labels == 1)).sum()
    old_to_new_1_to_0 = ((old_labels == 1) & (new_labels == 0)).sum()
    
    print(f"\nTipos de cambios:")
    print(f"  0→1 (No-Speech a Speech): {old_to_new_0_to_1}")
    print(f"  1→0 (Speech a No-Speech): {old_to_new_1_to_0}")
    
    # Matriz de confusión
    cm = confusion_matrix(old_labels, new_labels)
    
    print(f"\nMatriz de Confusión (Viejo vs Nuevo):")
    print(f"           Nuevo")
    print(f"Viejo     0     1")
    print(f"    0  {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"    1  {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Accuracy de la "predicción" nuevo vs viejo
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    print(f"\nAcuerdo entre GT: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Mostrar archivos que cambiaron
    changed_files = [common_sorted[i] for i in range(len(common_sorted)) if changes[i]]
    
    print(f"\nPrimeros 10 archivos que cambiaron:")
    for i, file in enumerate(changed_files[:10]):
        old_label = old_common.loc[file, 'Speech']
        new_label = new_common.loc[file, 'Speech']
        
        # Intentar obtener majority vote si disponible
        vote_info = ""
        if new_gt_with_votes is not None and not new_gt_with_votes.empty:
            vote_row = new_gt_with_votes[new_gt_with_votes['Chunk'] == file]
            if not vote_row.empty:
                majority_vote = vote_row.iloc[0].get('MajorityVote', 'N/A')
                vote_info = f" (vote: '{majority_vote}')"
        
        print(f"  {i+1}. {file}: {old_label}→{new_label}{vote_info}")
    
    return cm, changed_files

def generate_comparison_plots(old_gt: pd.DataFrame, new_gt: pd.DataFrame, 
                            common_files: set, scenario: str, output_dir: str):
    """Generar gráficos de comparación."""
    if old_gt.empty or new_gt.empty or not common_files:
        return
    
    # Filtrar a archivos comunes
    old_common = old_gt[old_gt['Chunk'].isin(common_files)].set_index('Chunk').sort_index()
    new_common = new_gt[new_gt['Chunk'].isin(common_files)].set_index('Chunk').sort_index()
    
    common_sorted = sorted(common_files)
    old_labels = old_common.loc[common_sorted, 'Speech'].values
    new_labels = new_common.loc[common_sorted, 'Speech'].values
    
    # Configurar matplotlib
    plt.style.use('default')
    plt.rcParams.update({'font.size': 10})
    
    # 1. Matriz de confusión
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    cm = confusion_matrix(old_labels, new_labels)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['No-Speech', 'Speech'],
                yticklabels=['No-Speech', 'Speech'])
    ax1.set_title(f'Confusion Matrix - {scenario.upper()}\n(Old GT vs New GT)')
    ax1.set_xlabel('New GT')
    ax1.set_ylabel('Old GT')
    
    # 2. Distribución de etiquetas
    old_dist = np.bincount(old_labels, minlength=2)
    new_dist = np.bincount(new_labels, minlength=2)
    
    x = np.arange(2)
    width = 0.35
    
    ax2.bar(x - width/2, old_dist, width, label='Old GT', alpha=0.8)
    ax2.bar(x + width/2, new_dist, width, label='New GT', alpha=0.8)
    
    ax2.set_title(f'Label Distribution - {scenario.upper()}')
    ax2.set_xlabel('Label')
    ax2.set_ylabel('Count')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['No-Speech', 'Speech'])
    ax2.legend()
    
    # Añadir valores en las barras
    for i, (old_val, new_val) in enumerate(zip(old_dist, new_dist)):
        ax2.text(i - width/2, old_val + max(old_dist)*0.01, str(old_val), 
                ha='center', va='bottom')
        ax2.text(i + width/2, new_val + max(new_dist)*0.01, str(new_val), 
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Guardar
    os.makedirs(output_dir, exist_ok=True)
    plot_file = os.path.join(output_dir, f'gt_comparison_{scenario}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📈 Gráfico guardado: {plot_file}")

def save_comparison_report(comparisons: dict, output_dir: str):
    """Guardar reporte detallado de comparación."""
    os.makedirs(output_dir, exist_ok=True)
    
    report_file = os.path.join(output_dir, "gt_comparison_report.txt")
    
    with open(report_file, 'w') as f:
        f.write("REPORTE DE COMPARACIÓN GROUND TRUTH\n")
        f.write("=" * 50 + "\n\n")
        
        for scenario, data in comparisons.items():
            f.write(f"ESCENARIO: {scenario.upper()}\n")
            f.write("-" * 30 + "\n")
            
            if 'basic_stats' in data:
                f.write("Estadísticas Básicas:\n")
                f.write(f"  GT Viejo - Total: {data['basic_stats']['old_total']}, Speech: {data['basic_stats']['old_speech']}\n")
                f.write(f"  GT Nuevo - Total: {data['basic_stats']['new_total']}, Speech: {data['basic_stats']['new_speech']}\n")
                f.write(f"  Diferencia Speech: {data['basic_stats']['new_speech'] - data['basic_stats']['old_speech']}\n\n")
            
            if 'overlap' in data:
                f.write("Análisis de Overlap:\n")
                f.write(f"  Archivos comunes: {len(data['overlap']['common'])}\n")
                f.write(f"  Solo en viejo: {len(data['overlap']['only_old'])}\n")
                f.write(f"  Solo en nuevo: {len(data['overlap']['only_new'])}\n\n")
            
            if 'changes' in data and data['changes']['cm'] is not None:
                cm = data['changes']['cm']
                accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
                f.write("Cambios de Etiquetas:\n")
                f.write(f"  Archivos con cambios: {len(data['changes']['changed_files'])}\n")
                f.write(f"  Acuerdo entre GT: {accuracy:.3f}\n")
                f.write(f"  Cambios 0→1: {cm[0,1]}\n")
                f.write(f"  Cambios 1→0: {cm[1,0]}\n\n")
            
            f.write("\n")
    
    print(f"   📄 Reporte guardado: {report_file}")

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal de comparación."""
    
    # Verificar directorios
    if not os.path.exists(OLD_GT_DIR):
        raise FileNotFoundError(f"❌ Directorio GT viejo no encontrado: {OLD_GT_DIR}")
    
    if not os.path.exists(NEW_GT_DIR):
        raise FileNotFoundError(f"❌ Directorio GT nuevo no encontrado: {NEW_GT_DIR}")
    
    scenarios = ['cmf', 'cmfv']
    comparisons = {}
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"🔍 ANALIZANDO ESCENARIO: {scenario.upper()}")
        print(f"{'='*60}")
        
        # Cargar GT files
        old_gt = load_gt_file(OLD_GT_DIR, scenario)
        new_gt = load_gt_file(NEW_GT_DIR, scenario)
        new_gt_with_votes = load_gt_with_votes(NEW_GT_DIR, scenario)
        
        if old_gt.empty and new_gt.empty:
            print(f"⚠️ Ambos GT están vacíos para {scenario}")
            continue
        
        scenario_data = {}
        
        # 1. Estadísticas básicas
        compare_gt_basic_stats(old_gt, new_gt, scenario)
        
        if not old_gt.empty and not new_gt.empty:
            scenario_data['basic_stats'] = {
                'old_total': len(old_gt),
                'old_speech': old_gt['Speech'].sum(),
                'new_total': len(new_gt),
                'new_speech': new_gt['Speech'].sum()
            }
        
        # 2. Análisis de overlap
        common_files, only_old, only_new = analyze_file_overlap(old_gt, new_gt, scenario)
        
        scenario_data['overlap'] = {
            'common': common_files or set(),
            'only_old': only_old or set(),
            'only_new': only_new or set()
        }
        
        # 3. Análisis de cambios de etiquetas
        if common_files:
            result = analyze_label_changes(old_gt, new_gt, common_files, 
                                         scenario, new_gt_with_votes)
            if result is None:
                scenario_data['changes'] = {'cm': None, 'changed_files': []}
            else:
                cm, changed_files = result
                scenario_data['changes'] = {
                    'cm': cm,
                    'changed_files': changed_files or []
                }
        else:
            scenario_data['changes'] = {'cm': None, 'changed_files': []}
        
        # 4. Generar gráficos
        generate_comparison_plots(old_gt, new_gt, common_files or set(), 
                                scenario, COMPARISON_OUTPUT)
        
        comparisons[scenario] = scenario_data
    
    # 5. Guardar reporte
    save_comparison_report(comparisons, COMPARISON_OUTPUT)
    
    # 6. Resumen final
    print(f"\n{'='*60}")
    print("🎯 RESUMEN FINAL DE COMPARACIÓN")
    print(f"{'='*60}")
    
    for scenario, data in comparisons.items():
        print(f"\n{scenario.upper()}:")
        
        if 'basic_stats' in data:
            old_speech = data['basic_stats']['old_speech']
            new_speech = data['basic_stats']['new_speech']
            diff = new_speech - old_speech
            print(f"  Diferencia en Speech: {diff:+d}")
        
        if 'overlap' in data:
            common_count = len(data['overlap']['common'])
            print(f"  Archivos comunes: {common_count}")
        
        if 'changes' in data and data['changes']['cm'] is not None:
            cm = data['changes']['cm']
            if cm.sum() > 0:
                accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
                print(f"  Acuerdo GT: {accuracy:.1%}")
                print(f"  Archivos que cambiaron: {len(data['changes']['changed_files'])}")
    
    print(f"\n✅ Análisis de comparación completado")
    print(f"📁 Resultados en: {COMPARISON_OUTPUT}")

if __name__ == "__main__":
    main()