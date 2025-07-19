#!/usr/bin/env python3
"""
VERSIÓN CORREGIDA: Ejecutar evaluación en todos los escenarios SIN inferencia duplicada
"""

import subprocess
import json
import yaml
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import shutil
import glob
import argparse


def load_unified_config(config_file: str = "config_all_scenarios.yaml"):
    """Cargar configuración unificada desde el archivo indicado."""
    with open(config_file) as f:
        return yaml.safe_load(f)


def create_scenario_config(base_config, scenario_name, scenario_info):
    """Crear configuración temporal para un escenario específico."""
    scenario_config = {
        'project': base_config['project'],
        'test_settings': base_config['test_settings'],
        'models': base_config['models'],
        'datasets': {
            scenario_name: {
                'chunks_path': scenario_info['chunks_path'],
                'ground_truth': scenario_info['ground_truth']
            }
        }
    }

    temp_config_file = f"temp_config_{scenario_name}.yaml"
    with open(temp_config_file, 'w') as f:
        yaml.dump(scenario_config, f, default_flow_style=False)
    return temp_config_file


def clean_temp_configs():
    """Limpiar archivos de configuración temporales."""
    for temp_file in glob.glob("temp_config_*.yaml"):
        Path(temp_file).unlink()
        print(f"🧹 Config temporal eliminado: {temp_file}")


def run_inference_once(base_config):
    """
    🔧 NUEVO: Ejecutar inferencia UNA SOLA VEZ para todos los modelos.
    Usa el primer escenario para hacer inferencia, luego reutiliza máscaras.
    """
    print(f"\n{'='*60}")
    print("🔬 EJECUTANDO INFERENCIA ÚNICA (TODOS LOS MODELOS)")
    print(f"{'='*60}")
    
    # Usar primer escenario para inferencia
    scenarios = base_config['scenarios']
    first_scenario = list(scenarios.keys())[0]
    first_info = scenarios[first_scenario]
    
    print(f"📋 Usando dataset: {first_scenario}")
    print(f"📁 Path: {first_info['chunks_path']}")
    
    # Crear config temporal con TODOS los modelos habilitados
    inference_config = {
        'project': base_config['project'],
        'test_settings': base_config['test_settings'],
        'models': base_config['models'],  # Todos los modelos
        'datasets': {
            first_scenario: {
                'chunks_path': first_info['chunks_path'],
                'ground_truth': first_info['ground_truth']
            }
        }
    }
    
    temp_config_file = "temp_config_inference_only.yaml"
    with open(temp_config_file, 'w') as f:
        yaml.dump(inference_config, f, default_flow_style=False)
    
    # Ejecutar SOLO inferencia (sin evaluación)
    cmd = [
        "python", "scripts/run_evaluation.py",
        "--config", temp_config_file,
        "--verbose"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Inferencia única completada")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en inferencia única: {e}")
        return False
    finally:
        # Limpiar config temporal
        Path(temp_config_file).unlink(missing_ok=True)


def evaluate_scenario(scenario_name, scenario_info, base_config):
    """
    🔧 NUEVO: Evaluar un escenario REUTILIZANDO máscaras existentes.
    NO hace inferencia, solo evaluación.
    """
    print(f"\n{'='*60}")
    print(f"📊 EVALUANDO: {scenario_name.upper()}")
    print(f"📋 {scenario_info['description']}")
    print(f"{'='*60}")

    # Directorio base unificado
    base_path = Path(base_config['project']['base_path'])
    results_dir = base_path / 'results'
    scenario_dir = results_dir / f"scenario_{scenario_name.lower()}"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear config temporal solo para evaluación
    scenario_config = {
        'project': base_config['project'],
        'test_settings': base_config['test_settings'],
        'models': base_config['models'],
        'datasets': {
            scenario_name: {
                'chunks_path': scenario_info['chunks_path'],
                'ground_truth': scenario_info['ground_truth']
            }
        }
    }

    temp_config_file = f"temp_config_eval_{scenario_name}.yaml"
    with open(temp_config_file, 'w') as f:
        yaml.dump(scenario_config, f, default_flow_style=False)

    # Ejecutar SOLO evaluación (las máscaras ya existen)
    cmd = [
        "python", "scripts/run_evaluation.py",
        "--config", temp_config_file,
        "--verbose"
    ]

    try:
        subprocess.run(cmd, check=True)
        print("✅ Evaluación completada exitosamente")

        # Mover resultados al directorio del escenario
        for metric_file in glob.glob(str(results_dir / "metrics_*.json")):
            filename = Path(metric_file).name
            shutil.move(metric_file, scenario_dir / filename)
        for plot_file in glob.glob(str(results_dir / "plot_*.png")):
            filename = Path(plot_file).name
            shutil.move(plot_file, scenario_dir / filename)

        print(f"📁 Resultados guardados en: {scenario_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error evaluando {scenario_name}: {e}")
        return False
    finally:
        # Limpiar config temporal
        Path(temp_config_file).unlink(missing_ok=True)


def load_scenario_results(scenario_name, base_path):
    """Cargar resultados JSON de un escenario."""
    results_dir = Path(base_path) / 'results' / f"scenario_{scenario_name.lower()}"
    scenario_results = {}
    for json_file in results_dir.glob("metrics_*.json"):
        model_name = json_file.stem.replace("metrics_", "")
        data = json.load(open(json_file))
        if data.get('f1'):
            idx = data['f1'].index(max(data['f1']))
            best_f1 = data['f1'][idx]
            best_thresh = data['thresholds'][idx]
        else:
            best_f1, best_thresh = 0.0, 0.0
        scenario_results[model_name] = {
            'best_f1': best_f1,
            'best_threshold': best_thresh,
            'all_metrics': data
        }
    return scenario_results


def generate_comparison_report(scenarios_info, base_path):
    """Generar reporte comparativo de todos los escenarios."""
    print(f"\n{'='*60}")
    print("📊 GENERANDO REPORTE COMPARATIVO")
    print(f"{'='*60}")

    all_results = {}
    for scenario_name in scenarios_info.keys():
        try:
            results = load_scenario_results(scenario_name, base_path)
            all_results[scenario_name] = results
            print(f"✅ Cargado: {scenario_name} ({len(results)} modelos)")
        except Exception as e:
            print(f"❌ Error cargando {scenario_name}: {e}")

    if not all_results:
        print("❌ No se pudieron cargar resultados")
        return

    comparison_data = []
    for scenario_name, scenario_results in all_results.items():
        desc = scenarios_info[scenario_name]['description']
        for model_name, metrics in scenario_results.items():
            comparison_data.append({
                'Scenario': scenario_name,
                'Description': desc,
                'Model': model_name.upper(),
                'Best_F1': metrics['best_f1'],
                'Best_Threshold': metrics['best_threshold']
            })

    df = pd.DataFrame(comparison_data)
    print(f"\n📋 TABLA COMPARATIVA:")
    print("="*100)
    pivot = df.pivot_table(index=['Scenario','Description'], columns='Model', values='Best_F1', fill_value=0)
    print(pivot.round(3))

    plt.figure(figsize=(15,10))
    scenarios = df['Scenario'].unique()
    models = df['Model'].unique()
    x = list(range(len(scenarios)))
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        vals = [df[(df['Scenario']==s) & (df['Model']==model)]['Best_F1'].iloc[0] if len(df[(df['Scenario']==s) & (df['Model']==model)]) > 0 else 0 for s in scenarios]
        plt.bar([pos + width*i for pos in x], vals, width, label=model)

    plt.xlabel('Escenario')
    plt.ylabel('Mejor F1-Score')
    plt.title('Comparación de Performance por Escenario y Modelo')
    plt.xticks([pos + width*(len(models)-1)/2 for pos in x], [f"{s}\n({scenarios_info[s]['description']})" for s in scenarios], rotation=45, ha='right')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(0,1.1)
    plt.tight_layout()

    # Guardar en directorio base unificado
    results_dir = Path(base_path) / 'results'
    plt.savefig(results_dir / 'comparison_all_scenarios.png', dpi=300)
    df.to_csv(results_dir / 'comparison_all_scenarios.csv', index=False)

    print(f"\n✅ Reporte generado:")
    print(f"📊 Gráfico: {results_dir}/comparison_all_scenarios.png")
    print(f"📋 Datos: {results_dir}/comparison_all_scenarios.csv")


def main():
    parser = argparse.ArgumentParser(description='Evaluación VAD - SIN Inferencia Duplicada')
    parser.add_argument('--config', default='config_all_scenarios.yaml',
                        help='Archivo de configuración YAML')
    parser.add_argument('--skip-inference', action='store_true',
                        help='Saltar inferencia (solo evaluación)')
    args = parser.parse_args()

    print("🚀 EVALUACIÓN VAD - INFERENCIA EFICIENTE")
    print("📋 Cargando configuración...")
    try:
        config = load_unified_config(args.config)
        scenarios = config['scenarios']
        base_path = config['project']['base_path']
        print(f"✅ {len(scenarios)} escenarios encontrados")
        enabled_models = [name for name, m in config['models'].items() if m.get('enabled', True)]
        print(f"🤖 Modelos habilitados: {', '.join(enabled_models)}")
        print(f"📁 Directorio base: {base_path}")
    except Exception as e:
        print(f"❌ Error cargando configuración: {e}")
        return

    successful = []
    try:
        # PASO 1: Inferencia UNA SOLA VEZ (si no se saltea)
        if not args.skip_inference:
            print("\n🔬 PASO 1: INFERENCIA ÚNICA")
            if not run_inference_once(config):
                print("❌ Error en inferencia única, abortando")
                return
        else:
            print("\n⏭️ SALTANDO INFERENCIA (usando máscaras existentes)")

        # PASO 2: Evaluar cada escenario REUTILIZANDO máscaras
        print(f"\n📊 PASO 2: EVALUACIÓN POR ESCENARIO")
        for scen, info in scenarios.items():
            if evaluate_scenario(scen, info, config):
                successful.append(scen)

    finally:
        clean_temp_configs()

    if successful:
        generate_comparison_report(scenarios, base_path)
        print(f"\n🎯 RESUMEN FINAL:")
        print(f"✅ Escenarios exitosos: {len(successful)}/{len(scenarios)}")
        for scen in successful:
            print(f"   • {scen}: {scenarios[scen]['description']}")
        print(f"📁 Resultados detallados: {base_path}/results/scenario_*/")
        print(f"📊 Reporte comparativo: {base_path}/results/comparison_all_scenarios.*")
    else:
        print("\n❌ No se completó ningún escenario exitosamente")

if __name__ == "__main__":
    main()