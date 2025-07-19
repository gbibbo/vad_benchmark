#!/usr/bin/env python3
"""
scripts/run_evaluation.py - Script maestro para evaluación completa de modelos VAD

Flujo:
1. Verificar si existe inferencia previa
2. Si no existe → Inferencia en N ejemplos (prototipo)
3. Evaluación contra ground truth con prints detallados
4. Guardar métricas y generar gráficos
5. Resumen final comparativo

Uso:
    python scripts/run_evaluation.py --config config_test.yaml --mode prototype
    python scripts/run_evaluation.py --config config_full.yaml --mode full
"""

import sys
import os
import yaml
import logging
import importlib
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import glob
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# Agregar proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class VADEvaluator:
    """Evaluador maestro para modelos VAD."""
    
    def __init__(self, config_path: str, verbose: bool = True):
        self.config = self._load_config(config_path)
        self.verbose = verbose
        self.base_path = Path(self.config['project']['base_path'])
        self.results_dir = self.base_path / 'results'
        #self.results_dir = self.base_path / 'results'
        self.results_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Diccionario de resultados
        self.all_results = {}
        
    def _load_config(self, config_path: str) -> dict:
        """Cargar configuración YAML."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """Configurar logging con archivo y consola."""
        log_file = self.results_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO if self.verbose else logging.WARNING,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🚀 Iniciando evaluación VAD - Log: {log_file}")
    
    def _get_audio_files_limited(self, dataset_config: dict, max_files: int) -> List[str]:
        """
        🔧 ARREGLADO: Obtener archivos que SÍ estén en el ground truth.
        
        En lugar de seleccionar archivos aleatorios del dataset,
        lee el ground truth y busca esos archivos específicos.
        """
        # Cargar ground truth para saber qué archivos buscar
        gt_path = dataset_config['ground_truth']
        chunks_path = self.base_path / dataset_config['chunks_path']
        
        if not chunks_path.exists():
            self.logger.warning(f"📁 Dataset no encontrado: {chunks_path}")
            return []
        
        if not os.path.exists(gt_path):
            self.logger.warning(f"📁 Ground truth no encontrado: {gt_path}")
            return []
        
        # Leer ground truth
        try:
            gt_df = pd.read_csv(gt_path)
            
            # Detectar formato de columnas
            if 'Chunk' in gt_df.columns:
                chunk_col = 'Chunk'
            elif 'Filename' in gt_df.columns:
                chunk_col = 'Filename'
            else:
                chunk_col = gt_df.columns[0]  # Primera columna
            
            # Obtener lista de archivos del GT
            gt_files = gt_df[chunk_col].tolist()
            
            self.logger.info(f"📋 Ground truth: {len(gt_files)} archivos disponibles")
            
            # Buscar los archivos correspondientes en chunks
            found_files = []
            searched_count = 0
            
            for gt_file in gt_files:
                searched_count += 1
                
                # Buscar archivo .wav correspondiente
                # Probar diferentes patrones
                patterns = [
                    gt_file,                    # NUEVO: Para MUSAN (path completo)
                    f"**/{gt_file}",           # NUEVO: MUSAN en subdirectorios
                    f"{gt_file}.16kHz.wav",    # ORIGINAL: Para CHiME  
                    f"{gt_file}.wav",          # ORIGINAL: Fallback
                    f"*{gt_file}*.wav",        # ORIGINAL: Wildcard
                    f"{gt_file}*.wav"          # ORIGINAL: Suffix wildcard
                ]
                
                file_found = False
                for pattern in patterns:
                    matches = list(chunks_path.glob(pattern))
                    if matches:
                        found_files.append(str(matches[0]))
                        if self.verbose and len(found_files) <= 5:  # Solo log primeros 5
                            self.logger.info(f"✅ {gt_file} → {matches[0].name}")
                        file_found = True
                        break
                
                if not file_found and self.verbose and searched_count <= 5:
                    self.logger.info(f"❌ {gt_file} → No encontrado")
                
                # Parar cuando tengamos suficientes archivos
                if len(found_files) >= max_files:
                    break
            
            if found_files:
                self.logger.info(f"📊 Dataset {chunks_path.name}: {len(found_files)}/{searched_count} archivos encontrados del GT")
            else:
                self.logger.warning(f"⚠️ No se encontraron archivos del GT en {chunks_path}")
                
                # Fallback: usar método anterior si no se encuentran archivos del GT
                self.logger.info("🔄 Usando selección aleatoria como fallback...")
                return self._get_audio_files_fallback(chunks_path, max_files)
            
            return found_files
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error leyendo GT {gt_path}: {e}")
            return self._get_audio_files_fallback(chunks_path, max_files)
    
    def _get_audio_files_fallback(self, dataset_path: Path, max_files: int) -> List[str]:
        """Método fallback de selección aleatoria (método original)."""
        patterns = ["**/*.wav", "**/*16kHz*.wav", "*/*.wav"]
        audio_files = []
        
        for pattern in patterns:
            files = glob.glob(str(dataset_path / pattern), recursive=True)
            if files:
                audio_files.extend(files)
                break
        
        limited_files = sorted(audio_files)[:max_files]
        self.logger.info(f"📊 Fallback: {len(audio_files)} archivos, usando {len(limited_files)}")
        
        return limited_files
    
    def _load_wrapper(self, wrapper_path: str):
        """Cargar wrapper dinámicamente con debug mejorado."""
        try:
            # Fix: No hacer rsplit, importar el path completo
            module_path = wrapper_path
            self.logger.info(f"🔍 Intentando importar módulo: {module_path}")
            
            module = importlib.import_module(module_path)
            self.logger.info(f"✅ Módulo importado correctamente")
            
            # Debug: mostrar todas las clases del módulo
            all_classes = []
            wrapper_classes = []
            
            for attr_name in dir(module):
                if not attr_name.startswith('_'):  # Skip private
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type):
                        all_classes.append(attr_name)
                        has_infer = hasattr(attr, 'infer')
                        if has_infer and attr_name != 'BaseVADWrapper':
                            wrapper_classes.append((attr_name, attr))
                            self.logger.info(f"  🎯 Clase wrapper encontrada: {attr_name}")
            
            self.logger.info(f"📋 Todas las clases en módulo: {all_classes}")
            
            if not wrapper_classes:
                self.logger.error(f"❌ No se encontraron clases con método 'infer' en {wrapper_path}")
                return None
            
            # Preferir clase que termine en 'Wrapper' o contenga 'VAD'
            for name, cls in wrapper_classes:
                if 'wrapper' in name.lower() or 'vad' in name.lower():
                    self.logger.info(f"🔧 Cargando wrapper preferido: {name}")
                    return cls()
            
            # Si no hay preferencia, usar la primera
            name, cls = wrapper_classes[0]
            self.logger.info(f"🔧 Cargando primer wrapper disponible: {name}")
            return cls()
            
        except Exception as e:
            import traceback
            self.logger.error(f"❌ Error cargando wrapper {wrapper_path}: {e}")
            self.logger.error(f"🔍 Traceback: {traceback.format_exc()}")
            return None
    
    def _is_binary_model(self, model_name: str) -> bool:
        """Detectar si un modelo es binario (devuelve solo 0/1) o probabilístico."""
        # Lista de modelos conocidos como binarios
        binary_models = [
            'webrtc', 'whisper', 'whisper_tiny', 'whisper_small'
        ]
        
        # Verificar si el nombre del modelo indica que es binario
        model_lower = model_name.lower()
        for binary_name in binary_models:
            if binary_name in model_lower:
                return True
        
        return False

    def _masks_exist(self, model_name: str) -> bool:
        """Verificar si existen máscaras para un modelo."""
        masks_dir = self.results_dir / f"masks_{model_name}"
        if not masks_dir.exists():
            return False
        
        mask_files = list(masks_dir.glob("mask_*.csv"))
        return len(mask_files) > 0
    
    def _run_inference(self, model_name: str, wrapper, audio_files: List[str], 
                    thresholds: List[float]) -> bool:
        """Ejecutar inferencia para un modelo en archivos limitados."""
        self.logger.info(f"🔬 Ejecutando inferencia {model_name.upper()} en {len(audio_files)} archivos...")
        
        masks_dir = self.results_dir / f"masks_{model_name}"
        masks_dir.mkdir(exist_ok=True)
        
        # Verificar si es modelo binario
        is_binary = self._is_binary_model(model_name)
        if is_binary:
            self.logger.info(f"   🎯 Modelo binario detectado: {model_name}")
        
        # Inicializar máscaras para cada threshold
        masks = {thresh: [] for thresh in thresholds}
        
        success_count = 0
        all_probs = []  # Para análisis de distribución
        
        for i, audio_path in enumerate(audio_files):
            filename = os.path.basename(audio_path)
            
            try:
                self.logger.info(f"  📄 {i+1}/{len(audio_files)}: {filename}")
                
                # Inferencia
                probs = wrapper.infer(audio_path)
                # Soportar tensores PyTorch, ndarrays o listas
                if isinstance(probs, torch.Tensor):
                    max_prob = probs.max().item()
                else:
                    max_prob = float(np.max(probs)) if len(probs) > 0 else 0.0
                all_probs.append(max_prob)
                
                # Aplicar thresholds
                if is_binary:
                    # Para modelos binarios: usar decisión directa, ignorar thresholds
                    has_speech_binary = int(max_prob > 0)
                    for threshold in thresholds:
                        masks[threshold].append([filename, has_speech_binary])
                        
                    self.logger.info(f"    ✅ {len(probs)} frames, binario={has_speech_binary}")
                else:
                    # Para modelos probabilísticos: aplicar thresholds normalmente
                    for threshold in thresholds:
                        has_speech = int(max_prob >= threshold)
                        masks[threshold].append([filename, has_speech])
                    
                    self.logger.info(f"    ✅ {len(probs)} frames, max_prob={max_prob:.3f}")
                
                success_count += 1
                
            except Exception as e:
                self.logger.error(f"    ❌ Error: {e}")
                # Agregar como no-speech en caso de error
                for threshold in thresholds:
                    masks[threshold].append([filename, 0])
        
        # Análisis de distribución para debug
        if all_probs:
            unique_probs = np.unique(all_probs)
            self.logger.info(f"   📊 Distribución de probabilidades: {unique_probs}")
            
            if is_binary and len(unique_probs) > 2:
                self.logger.warning(f"   ⚠️  Modelo marcado como binario pero tiene {len(unique_probs)} valores únicos")
        
        # Guardar máscaras
        for threshold, rows in masks.items():
            mask_file = masks_dir / f"mask_{threshold:.2f}.csv"
            df = pd.DataFrame(rows, columns=['Filename', 'Speech'])
            df.to_csv(mask_file, index=False)
        
        self.logger.info(f"✅ Inferencia completada: {success_count}/{len(audio_files)} exitosos")
        return success_count > 0   
    
    def _load_ground_truth(self, gt_path: str) -> pd.DataFrame:
        """Cargar ground truth CSV - flexible para diferentes formatos."""
        df = pd.read_csv(gt_path)

        # ------------------------------------------------------------------
        # 1. Detectar formato de columnas  ─────────────
        #    · CHiME      → (Chunk, Condition)           normal
        #    · CHiME 2    → (Filename, Speech)           normal
        #    · MUSAN puro → 1 sola columna con la ruta   deducir Speech
        # ------------------------------------------------------------------
        filename_col = label_col = None

        if {'Chunk', 'Condition'}.issubset(df.columns):
            filename_col, label_col = 'Chunk', 'Condition'

        elif {'Filename', 'Speech'}.issubset(df.columns):
            filename_col, label_col = 'Filename', 'Speech'

        else:
            # ---- Caso MUSAN: sin etiqueta explícita ----------------------
            # Esperamos una única columna con la ruta relativa al corpus.
            if df.shape[1] == 1:
                filename_col = df.columns[0]

                # Deducir Speech (=1) ó no-speech (=0) según la carpeta
                df['Speech'] = df[filename_col].apply(
                    lambda p: 1 if '/speech/' in str(p).lower() else 0)

                label_col = 'Speech'
                self.logger.warning(
                    "MUSAN GT sin columna de etiqueta: 'Speech' deducido "
                    "por la carpeta (speech/music/noise).")
            else:
                raise ValueError(
                    f"Ground truth debe tener columnas (Chunk,Condition) "
                    f"o (Filename,Speech). Encontrado: {list(df.columns)}")

        # ------------------------------------------------------------------
        # 2. Normalización del nombre de archivo (común a todos los casos)
        # ------------------------------------------------------------------
        import re                                    # <-- necesario una vez
        def normalize_filename(filename):
            filename = os.path.basename(filename)
            # elimina sufijos de tasa + extensión
            filename = re.sub(r'\.(16|48)kHz(\.wav)?$', '', filename,
                              flags=re.IGNORECASE)
            filename = re.sub(r'\.wav$', '', filename, flags=re.IGNORECASE)
            return filename

        df['Filename_norm'] = df[filename_col].apply(normalize_filename)
        df['Label'] = df[label_col]
        
        self.logger.info(f"📋 Ground truth: {len(df)} archivos, columnas: {filename_col}, {label_col}")
        
        # Debug: mostrar algunos ejemplos de normalización
        if self.verbose and len(df) > 0:
            self.logger.info("🔍 Ejemplos de normalización:")
            for i in range(min(3, len(df))):
                orig = df[filename_col].iloc[i]
                norm = df['Filename_norm'].iloc[i]
                self.logger.info(f"  '{orig}' → '{norm}'")
        
        return df[['Filename_norm', 'Label']]
    
    def _evaluate_model(self, model_name: str, gt_df: pd.DataFrame) -> Dict:
        """Evaluar un modelo contra ground truth."""
        self.logger.info(f"📊 Evaluando {model_name.upper()} contra ground truth...")
        
        masks_dir = self.results_dir / f"masks_{model_name}"
        mask_files = sorted(masks_dir.glob("mask_*.csv"))
        
        if not mask_files:
            self.logger.warning(f"❌ No se encontraron máscaras para {model_name}")
            return {}
        
        results = {
            'thresholds': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'accuracy': [],
            'predictions': {},  # Para debugging
        }
        
        # 🔧 ARREGLADO: Contador de matches para debugging
        total_matches = 0
        
        for mask_file in mask_files:
            # Extraer threshold del nombre del archivo
            threshold = float(mask_file.stem.split('_')[1])
            
            # Cargar predicciones
            pred_df = pd.read_csv(mask_file)
            
            # Normalización robusta para predicciones también
            def normalize_prediction_filename(filename):
                """Misma normalización robusta que en ground truth."""
                filename = os.path.basename(filename)
                
                extensions_to_remove = [
                    '.16kHz.wav', '.48kHz.wav', '.16kHz.WAV', '.48kHz.WAV',
                    '.16khz.wav', '.48khz.wav',
                    '.16kHz',     '.48kHz',     '.16khz',     '.48khz',
                    '.wav', '.WAV', '.mp3', '.MP3', '.flac', '.FLAC'
                ]
                
                filename_lower = filename.lower()
                for ext in extensions_to_remove:
                    if filename_lower.endswith(ext.lower()):
                        filename = filename[:-len(ext)]
                        break
                
                return filename
            
            pred_df['Filename_norm'] = pred_df['Filename'].apply(normalize_prediction_filename)
            
            # 🔧 ARREGLADO: Debug del matching process
            if threshold == 0.5 and self.verbose:  # Solo para threshold principal
                self.logger.info(f"🔍 Debug matching para threshold {threshold}:")
                self.logger.info(f"   GT tiene {len(gt_df)} archivos")
                self.logger.info(f"   Predicciones tiene {len(pred_df)} archivos")
            
            # Merge con ground truth
            merged = pd.merge(gt_df, pred_df, on='Filename_norm', how='inner')
            
            if merged.empty:
                self.logger.warning(f"⚠️  No hay matches para threshold {threshold}")
                continue
            else:
                matches_count = len(merged)
                total_matches += matches_count
                if threshold == 0.5:  # Log solo para threshold principal
                    self.logger.info(f"✅ {matches_count} matches encontrados para threshold {threshold}")
            
            # Calcular métricas
            y_true = merged['Label'].astype(int)
            y_pred = merged['Speech'].astype(int)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            accuracy = (y_true == y_pred).mean()
            
            results['thresholds'].append(threshold)
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['f1'].append(f1)
            results['accuracy'].append(accuracy)
            
            # Guardar algunas predicciones para debugging
            if threshold == 0.5:  # Solo para threshold principal
                results['predictions'][threshold] = merged[['Filename_norm', 'Label', 'Speech']].to_dict('records')
        
        # 🔧 ARREGLADO: Log resumen de matches
        if total_matches > 0:
            self.logger.info(f"📈 Total matches encontrados: {total_matches} (across all thresholds)")
        else:
            self.logger.warning(f"⚠️ NO se encontraron matches en ningún threshold")
        
        return results
    
    def _print_predictions_vs_gt(self, model_name: str, predictions: List[Dict]):
        """Imprimir predicciones vs ground truth para debugging."""
        if not predictions:
            return
            
        self.logger.info(f"\n🔍 PREDICCIONES vs GROUND TRUTH - {model_name.upper()}")
        self.logger.info("="*70)
        self.logger.info(f"{'Archivo':<30} {'GT':<5} {'Pred':<5} {'Correcto':<10}")
        self.logger.info("-"*70)
        
        correct = 0
        for pred in predictions[:10]:  # Solo primeros 10
            filename = pred['Filename_norm'][:27] + "..." if len(pred['Filename_norm']) > 30 else pred['Filename_norm']
            gt = pred['Label']
            prediction = pred['Speech']
            is_correct = "✅ SÍ" if gt == prediction else "❌ NO"
            
            if gt == prediction:
                correct += 1
                
            self.logger.info(f"{filename:<30} {gt:<5} {prediction:<5} {is_correct:<10}")
        
        accuracy = correct / len(predictions[:10])
        self.logger.info("-"*70)
        self.logger.info(f"Precisión en muestra: {accuracy:.2%} ({correct}/{len(predictions[:10])})")
        self.logger.info("="*70)
    
    def _save_results(self, model_name: str, results: Dict):
        """Guardar resultados en JSON."""
        results_file = self.results_dir / f"metrics_{model_name}.json"
        
        # Preparar datos para JSON (sin las predicciones detalladas)
        json_results = {k: v for k, v in results.items() if k != 'predictions'}
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"💾 Métricas guardadas: {results_file}")
    
    def _generate_plots(self, model_name: str, results: Dict):
        """Generar gráficos de performance."""
        if not results.get('thresholds'):
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        thresholds = results['thresholds']
        
        # 1. Precision vs Threshold
        ax1.plot(thresholds, results['precision'], 'b-o', label='Precision')
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Precision')
        ax1.set_title(f'{model_name.upper()} - Precision vs Threshold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 2. Recall vs Threshold
        ax2.plot(thresholds, results['recall'], 'g-o', label='Recall')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Recall')
        ax2.set_title(f'{model_name.upper()} - Recall vs Threshold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # 3. F1 vs Threshold
        ax3.plot(thresholds, results['f1'], 'r-o', label='F1-Score')
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('F1-Score')
        ax3.set_title(f'{model_name.upper()} - F1-Score vs Threshold')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # 4. Todas las métricas juntas
        ax4.plot(thresholds, results['precision'], 'b-o', label='Precision', alpha=0.7)
        ax4.plot(thresholds, results['recall'], 'g-o', label='Recall', alpha=0.7)
        ax4.plot(thresholds, results['f1'], 'r-o', label='F1-Score', alpha=0.7)
        ax4.plot(thresholds, results['accuracy'], 'm-o', label='Accuracy', alpha=0.7)
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('Score')
        ax4.set_title(f'{model_name.upper()} - Todas las Métricas')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Guardar gráfico
        plot_file = self.results_dir / f"plot_{model_name}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📈 Gráfico guardado: {plot_file}")
    
    def _generate_comparison_plot(self):
        """Generar gráfico comparativo de todos los modelos."""
        if len(self.all_results) < 2:
            return
            
        plt.figure(figsize=(15, 10))
        
        for model_name, results in self.all_results.items():
            if not results.get('thresholds'):
                continue
                
            plt.plot(results['thresholds'], results['f1'], 
                    'o-', label=f'{model_name.upper()}', linewidth=2, markersize=6)
        
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('F1-Score', fontsize=12)
        plt.title('Comparación F1-Score por Modelo', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Guardar gráfico comparativo
        comparison_plot = self.results_dir / "comparison_all_models.png"
        plt.savefig(comparison_plot, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 Gráfico comparativo guardado: {comparison_plot}")
    
    def _print_final_summary(self):
        """Imprimir resumen final de todos los modelos."""
        if not self.all_results:
            return
            
        self.logger.info(f"\n🎯 RESUMEN FINAL - COMPARACIÓN DE MODELOS")
        self.logger.info("="*80)
        self.logger.info(f"{'Modelo':<15} {'Mejor F1':<12} {'@ Threshold':<12} {'Mejor Precisión':<15} {'Mejor Recall':<12}")
        self.logger.info("-"*80)
        
        for model_name, results in self.all_results.items():
            if not results.get('f1'):
                continue
                
            best_f1_idx = np.argmax(results['f1'])
            best_f1 = results['f1'][best_f1_idx]
            best_f1_thresh = results['thresholds'][best_f1_idx]
            
            best_precision = max(results['precision'])
            best_recall = max(results['recall'])
            
            self.logger.info(f"{model_name.upper():<15} {best_f1:<12.3f} {best_f1_thresh:<12.2f} "
                           f"{best_precision:<15.3f} {best_recall:<12.3f}")
        
        self.logger.info("="*80)
    
    def run_evaluation(self):
        """Ejecutar evaluación completa."""
        start_time = time.time()
        
        # Configuración
        max_files = self.config['test_settings']['max_files']
        thresholds = self.config['test_settings']['thresholds']
        
        # Buscar dataset disponible
        dataset_name = None
        audio_files = []
        gt_path = None
        datasets_dict = (self.config.get('datasets') or self.config.get('scenarios'))
        if datasets_dict is None:
            raise KeyError("La configuración debe tener una clave 'datasets' "
                        "o 'scenarios' al nivel raíz.")
        for name, dataset_config in datasets_dict.items():
            dataset_path = self.base_path / dataset_config['chunks_path']
            if dataset_path.exists():
                audio_files = self._get_audio_files_limited(dataset_config, max_files)
                gt_path = dataset_config['ground_truth']
                dataset_name = name
                break
        
        if not audio_files:
            self.logger.error("❌ No se encontraron archivos de audio")
            return
        
        # Cargar ground truth
        if not os.path.exists(gt_path):
            self.logger.error(f"❌ Ground truth no encontrado: {gt_path}")
            return
            
        gt_df = self._load_ground_truth(gt_path)
        self.logger.info(f"📋 Ground truth cargado: {len(gt_df)} archivos")
        
        # Evaluar cada modelo
        for model_name, model_config in self.config['models'].items():
            if not model_config.get('enabled', True):
                self.logger.info(f"⏭️  Modelo {model_name} deshabilitado, saltando...")
                continue
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🤖 PROCESANDO MODELO: {model_name.upper()}")
            self.logger.info(f"{'='*60}")
            
            # 1. Verificar si existe inferencia previa
            if self._masks_exist(model_name):
                self.logger.info(f"✅ Máscaras encontradas para {model_name}, saltando inferencia")
            else:
                # 2. Cargar wrapper y hacer inferencia
                wrapper = self._load_wrapper(model_config['wrapper'])
                if wrapper is None:
                    self.logger.error(f"❌ No se pudo cargar wrapper para {model_name}")
                    continue
                
                self.logger.info(f"✅ Wrapper cargado: {wrapper}")
                
                success = self._run_inference(model_name, wrapper, audio_files, thresholds)
                if not success:
                    self.logger.error(f"❌ Inferencia falló para {model_name}")
                    continue
            
            # 3. Evaluar contra ground truth
            results = self._evaluate_model(model_name, gt_df)
            if not results:
                continue
            
            # 4. Mostrar predicciones vs GT para debugging
            if 0.5 in results.get('predictions', {}):
                self._print_predictions_vs_gt(model_name, results['predictions'][0.5])
            
            # 5. Guardar resultados
            self._save_results(model_name, results)
            
            # 6. Generar gráficos
            self._generate_plots(model_name, results)
            
            # Guardar para resumen final
            self.all_results[model_name] = results
            
            # Log métricas principales
            if results.get('f1'):
                best_f1_idx = np.argmax(results['f1'])
                best_f1 = results['f1'][best_f1_idx]
                best_thresh = results['thresholds'][best_f1_idx]
                self.logger.info(f"🎯 Mejor F1: {best_f1:.3f} @ threshold {best_thresh:.2f}")
        
        # 7. Gráfico comparativo y resumen final
        self._generate_comparison_plot()
        self._print_final_summary()
        
        total_time = time.time() - start_time
        self.logger.info(f"\n✅ Evaluación completada en {total_time:.1f} segundos")
        self.logger.info(f"📁 Resultados guardados en: {self.results_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluación completa de modelos VAD')
    parser.add_argument('--config', default='config_test.yaml', 
                       help='Archivo de configuración YAML')
    parser.add_argument('--verbose', action='store_true', 
                       help='Mostrar output detallado')
    parser.add_argument('--mode', choices=['prototype', 'full'], default='prototype',
                       help='Modo de evaluación (prototype: pocos archivos, full: todos)')
    
    args = parser.parse_args()
    
    # Crear evaluador y ejecutar
    evaluator = VADEvaluator(args.config, verbose=args.verbose)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()