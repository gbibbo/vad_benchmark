#!/usr/-bin/env python3
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
# --- VAD-Benchmark sys.path guard ---
# --- Import hygiene for 'models' shadowing ---
import sys
if 'models' in sys.modules and not getattr(sys.modules['models'], '__path__', None):
    del sys.modules['models']
# --- end hygiene ---
"""!
@file run_evaluation.py
@brief Master script for the complete evaluation of VAD models.

@section workflow Workflow
1.  Check if previous inference results exist.
2.  If not, run inference on N examples (prototype).
3.  Evaluate against ground truth with detailed printouts.
4.  Save metrics and generate plots.
5.  Provide a final comparative summary.

@section usage Usage
@code
python scripts/run_evaluation.py --config config_test.yaml --mode prototype
python scripts/run_evaluation.py --config config_full.yaml --mode full
@endcode
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

# Add project to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class VADEvaluator:
    """! @brief Master evaluator for VAD models. """
    
    def __init__(self, config_path: str, verbose: bool = True):
        """!
        @brief Initializes the VADEvaluator.
        @param config_path Path to the YAML configuration file.
        @param verbose If True, enables detailed logging to the console.
        """
        self.config = self._load_config(config_path)
        self.verbose = verbose
        self.base_path = Path(self.config['project']['base_path'])
        self.results_dir = self.base_path / 'results'
        self.results_dir.mkdir(exist_ok=True)
        
        self._setup_logging()
        
        # Results dictionary
        self.all_results = {}
        
    def _load_config(self, config_path: str) -> dict:
        """!
        @brief Loads the YAML configuration file.
        @param config_path Path to the YAML configuration file.
        @return A dictionary with the loaded configuration.
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """!
        @brief Sets up logging to both a file and the console.
        """
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
        self.logger.info(f"🚀 Starting VAD evaluation - Log: {log_file}")
    
    def _get_audio_files_limited(self, dataset_config: dict, max_files: int) -> List[str]:
        """!
        @brief Gets a limited list of audio files that are present in the ground truth.
        
        @note [FIXED] Instead of selecting random files from the dataset, this method
              reads the ground truth and searches for those specific files.
              
        @param dataset_config The configuration dictionary for the dataset.
        @param max_files The maximum number of files to return.
        @return A list of paths to the audio files.
        """
        # Load ground truth to know which files to look for
        gt_path = dataset_config['ground_truth']
        chunks_path = self.base_path / dataset_config['chunks_path']
        
        if not chunks_path.exists():
            self.logger.warning(f"📁 Dataset not found: {chunks_path}")
            return []
        
        if not os.path.exists(gt_path):
            self.logger.warning(f"📁 Ground truth not found: {gt_path}")
            return []
        
        # Read ground truth
        try:
            gt_df = pd.read_csv(gt_path)
            
            # Detect column format
            if 'Chunk' in gt_df.columns:
                chunk_col = 'Chunk'
            elif 'Filename' in gt_df.columns:
                chunk_col = 'Filename'
            else:
                chunk_col = gt_df.columns[0]  # First column
            
            # Get list of files from the GT
            gt_files = gt_df[chunk_col].tolist()
            
            self.logger.info(f"📋 Ground truth: {len(gt_files)} files available")
            
            # Search for the corresponding files in chunks
            found_files = []
            searched_count = 0
            
            for gt_file in gt_files:
                searched_count += 1
                
                # Search for the corresponding .wav file
                # Try different patterns
                patterns = [
                    gt_file,                    # NEW: For MUSAN (full path)
                    f"**/{gt_file}",           # NEW: MUSAN in subdirectories
                    f"{gt_file}.16kHz.wav",    # ORIGINAL: For CHiME  
                    f"{gt_file}.wav",          # ORIGINAL: Fallback
                    f"*{gt_file}*.wav",        # ORIGINAL: Wildcard
                    f"{gt_file}*.wav"          # ORIGINAL: Suffix wildcard
                ]
                
                file_found = False
                for pattern in patterns:
                    matches = list(chunks_path.glob(pattern))
                    if matches:
                        found_files.append(str(matches[0]))
                        if self.verbose and len(found_files) <= 5:  # Only log the first 5
                            self.logger.info(f"✅ {gt_file} → {matches[0].name}")
                        file_found = True
                        break
                
                if not file_found and self.verbose and searched_count <= 5:
                    self.logger.info(f"❌ {gt_file} → Not found")
                
                # Stop when we have enough files
                if len(found_files) >= max_files:
                    break
            
            if found_files:
                self.logger.info(f"📊 Dataset {chunks_path.name}: {len(found_files)}/{searched_count} files from GT found")
            else:
                self.logger.warning(f"⚠️ No files from GT were found in {chunks_path}")
                
                # Fallback: use the previous method if GT files are not found
                self.logger.info("🔄 Using random selection as fallback...")
                return self._get_audio_files_fallback(chunks_path, max_files)
            
            return found_files
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error reading GT {gt_path}: {e}")
            return self._get_audio_files_fallback(chunks_path, max_files)
    
    def _get_audio_files_fallback(self, dataset_path: Path, max_files: int) -> List[str]:
        """!
        @brief Fallback method for random file selection (original method).
        @param dataset_path The path to the dataset directory.
        @param max_files The maximum number of files to select.
        @return A list of paths to the audio files.
        """
        patterns = ["**/*.wav", "**/*16kHz*.wav", "*/*.wav"]
        audio_files = []
        
        for pattern in patterns:
            files = glob.glob(str(dataset_path / pattern), recursive=True)
            if files:
                audio_files.extend(files)
                break
        
        limited_files = sorted(audio_files)[:max_files]
        self.logger.info(f"📊 Fallback: {len(audio_files)} files found, using {len(limited_files)}")
        
        return limited_files
    
    def _load_wrapper(self, wrapper_path: str):
        """!
        @brief Dynamically loads a VAD wrapper with enhanced debugging.
        @param wrapper_path The module path to the wrapper class.
        @return An instance of the wrapper class, or None on error.
        """
        try:
            # Fix: Do not rsplit, import the full path
            module_path = wrapper_path
            self.logger.info(f"🔍 Attempting to import module: {module_path}")
            
            module = importlib.import_module(module_path)
            self.logger.info(f"✅ Module imported successfully")
            
            # Debug: show all classes in the module
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
                            self.logger.info(f"  🎯 Wrapper class found: {attr_name}")
            
            self.logger.info(f"📋 All classes in module: {all_classes}")
            
            if not wrapper_classes:
                self.logger.error(f"❌ No classes with an 'infer' method found in {wrapper_path}")
                return None
            
            # Prefer a class ending in 'Wrapper' or containing 'VAD'
            for name, cls in wrapper_classes:
                if 'wrapper' in name.lower() or 'vad' in name.lower():
                    self.logger.info(f"🔧 Loading preferred wrapper: {name}")
                    return cls()
            
            # If no preference, use the first one found
            name, cls = wrapper_classes[0]
            self.logger.info(f"🔧 Loading first available wrapper: {name}")
            return cls()
            
        except Exception as e:
            import traceback
            self.logger.error(f"❌ Error loading wrapper {wrapper_path}: {e}")
            self.logger.error(f"🔍 Traceback: {traceback.format_exc()}")
            return None
    
    def _is_binary_model(self, model_name: str) -> bool:
        """!
        @brief Detects if a model is binary (returns only 0/1) or probabilistic.
        @param model_name The name of the model.
        @return True if the model is considered binary, False otherwise.
        """
        # List of models known to be binary
        binary_models = [
            'webrtc', 'whisper', 'whisper_tiny', 'whisper_small'
        ]
        
        # Check if the model name indicates it is binary
        model_lower = model_name.lower()
        for binary_name in binary_models:
            if binary_name in model_lower:
                return True
        
        return False

    def _masks_exist(self, model_name: str) -> bool:
        """!
        @brief Checks if inference masks already exist for a given model.
        @param model_name The name of the model.
        @return True if mask files are found, False otherwise.
        """
        masks_dir = self.results_dir / f"masks_{model_name}"
        if not masks_dir.exists():
            return False
        
        mask_files = list(masks_dir.glob("mask_*.csv"))
        return len(mask_files) > 0
    
    def _run_inference(self, model_name: str, wrapper, audio_files: List[str], 
                    thresholds: List[float]) -> bool:
        """!
        @brief Runs inference for a given model on a limited set of audio files.
        @param model_name The name of the model being evaluated.
        @param wrapper The loaded VAD model wrapper instance.
        @param audio_files A list of paths to the audio files for inference.
        @param thresholds A list of thresholds to apply to the model's output probabilities.
        @return True if the inference process was successful for at least one file, False otherwise.
        """
        self.logger.info(f"🔬 Running inference for {model_name.upper()} on {len(audio_files)} files...")
        
        masks_dir = self.results_dir / f"masks_{model_name}"
        masks_dir.mkdir(exist_ok=True)
        
        # Check if it's a binary model
        is_binary = self._is_binary_model(model_name)
        if is_binary:
            self.logger.info(f"   🎯 Binary model detected: {model_name}")
        
        # Initialize masks for each threshold
        masks = {thresh: [] for thresh in thresholds}
        
        success_count = 0
        all_probs = []  # For distribution analysis
        
        for i, audio_path in enumerate(audio_files):
            filename = os.path.basename(audio_path)
            
            try:
                self.logger.info(f"  📄 {i+1}/{len(audio_files)}: {filename}")
                
                # Inference
                probs = wrapper.infer(audio_path)
                # Support PyTorch tensors, ndarrays, or lists
                if isinstance(probs, torch.Tensor):
                    max_prob = probs.max().item()
                else:
                    max_prob = float(np.max(probs)) if len(probs) > 0 else 0.0
                all_probs.append(max_prob)
                
                # Apply thresholds
                if is_binary:
                    # For binary models: use the direct decision, ignore thresholds
                    has_speech_binary = int(max_prob > 0)
                    for threshold in thresholds:
                        masks[threshold].append([filename, has_speech_binary])
                        
                    self.logger.info(f"    ✅ {len(probs)} frames, binary={has_speech_binary}")
                else:
                    # For probabilistic models: apply thresholds normally
                    for threshold in thresholds:
                        has_speech = int(max_prob >= threshold)
                        masks[threshold].append([filename, has_speech])
                    
                    self.logger.info(f"    ✅ {len(probs)} frames, max_prob={max_prob:.3f}")
                
                success_count += 1
                
            except Exception as e:
                self.logger.error(f"    ❌ Error: {e}")
                # Add as non-speech in case of error
                for threshold in thresholds:
                    masks[threshold].append([filename, 0])
        
        # Distribution analysis for debugging
        if all_probs:
            unique_probs = np.unique(all_probs)
            self.logger.info(f"   📊 Probability distribution: {unique_probs}")
            
            if is_binary and len(unique_probs) > 2:
                self.logger.warning(f"   ⚠️  Model marked as binary but has {len(unique_probs)} unique values")
        
        # Save masks
        for threshold, rows in masks.items():
            mask_file = masks_dir / f"mask_{threshold:.2f}.csv"
            df = pd.DataFrame(rows, columns=['Filename', 'Speech'])
            df.to_csv(mask_file, index=False)
        
        self.logger.info(f"✅ Inference completed: {success_count}/{len(audio_files)} successful")
        return success_count > 0   
    
    def _load_ground_truth(self, gt_path: str) -> pd.DataFrame:
        """!
        @brief Loads the ground truth CSV file, handling different formats flexibly.
        @param gt_path Path to the ground truth CSV file.
        @return A pandas DataFrame with normalized filenames and labels.
        """
        df = pd.read_csv(gt_path)

        # ------------------------------------------------------------------
        # 1. Detect column format
        #    · CHiME      → (Chunk, Condition)           (normal)
        #    · CHiME 2    → (Filename, Speech)           (normal)
        #    · Pure MUSAN → 1 single column with the path   (deduce Speech)
        # ------------------------------------------------------------------
        filename_col = label_col = None

        if {'Chunk', 'Condition'}.issubset(df.columns):
            filename_col, label_col = 'Chunk', 'Condition'

        elif {'Filename', 'Speech'}.issubset(df.columns):
            filename_col, label_col = 'Filename', 'Speech'

        else:
            # ---- MUSAN case: no explicit label ---------------------------
            # Expect a single column with the relative path to the corpus.
            if df.shape[1] == 1:
                filename_col = df.columns[0]

                # Deduce Speech (=1) or non-speech (=0) based on the folder
                df['Speech'] = df[filename_col].apply(
                    lambda p: 1 if '/speech/' in str(p).lower() else 0)

                label_col = 'Speech'
                self.logger.warning(
                    "MUSAN GT has no label column: 'Speech' was inferred "
                    "from the folder name (speech/music/noise).")
            else:
                raise ValueError(
                    f"Ground truth must have columns (Chunk,Condition) "
                    f"or (Filename,Speech). Found: {list(df.columns)}")

        # ------------------------------------------------------------------
        # 2. Filename normalization (common to all cases)
        # ------------------------------------------------------------------
        import re
        def normalize_filename(filename):
            filename = os.path.basename(filename)
            # removes rate suffixes + extension
            filename = re.sub(r'\.(16|48)kHz(\.wav)?$', '', filename,
                              flags=re.IGNORECASE)
            filename = re.sub(r'\.wav$', '', filename, flags=re.IGNORECASE)
            return filename

        df['Filename_norm'] = df[filename_col].apply(normalize_filename)
        df['Label'] = df[label_col]
        
        self.logger.info(f"📋 Ground truth: {len(df)} files, columns: {filename_col}, {label_col}")
        
        # Debug: show some normalization examples
        if self.verbose and len(df) > 0:
            self.logger.info("🔍 Normalization examples:")
            for i in range(min(3, len(df))):
                orig = df[filename_col].iloc[i]
                norm = df['Filename_norm'].iloc[i]
                self.logger.info(f"  '{orig}' → '{norm}'")
        
        return df[['Filename_norm', 'Label']]
    
    def _evaluate_model(self, model_name: str, gt_df: pd.DataFrame) -> Dict:
        """!
        @brief Evaluates a model's predictions against the ground truth.
        @param model_name The name of the model.
        @param gt_df The pandas DataFrame containing the ground truth.
        @return A dictionary containing evaluation metrics (precision, recall, f1, etc.).
        """
        self.logger.info(f"📊 Evaluating {model_name.upper()} against ground truth...")
        
        masks_dir = self.results_dir / f"masks_{model_name}"
        mask_files = sorted(masks_dir.glob("mask_*.csv"))
        
        if not mask_files:
            self.logger.warning(f"❌ No masks found for {model_name}")
            return {}
        
        results = {
            'thresholds': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'accuracy': [],
            'predictions': {},  # For debugging
        }
        
        # 🔧 FIXED: Match counter for debugging
        total_matches = 0
        
        for mask_file in mask_files:
            # Extract threshold from the filename
            threshold = float(mask_file.stem.split('_')[1])
            
            # Load predictions
            pred_df = pd.read_csv(mask_file)
            
            # Robust normalization for predictions as well
            def normalize_prediction_filename(filename):
                """! Same robust normalization as in the ground truth."""
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
            
            # 🔧 FIXED: Debug the matching process
            if threshold == 0.5 and self.verbose:  # Only for the main threshold
                self.logger.info(f"🔍 Debugging match process for threshold {threshold}:")
                self.logger.info(f"   GT has {len(gt_df)} files")
                self.logger.info(f"   Predictions have {len(pred_df)} files")
            
            # Merge with ground truth
            merged = pd.merge(gt_df, pred_df, on='Filename_norm', how='inner')
            
            if merged.empty:
                self.logger.warning(f"⚠️  No matches found for threshold {threshold}")
                continue
            else:
                matches_count = len(merged)
                total_matches += matches_count
                if threshold == 0.5:  # Log only for the main threshold
                    self.logger.info(f"✅ {matches_count} matches found for threshold {threshold}")
            
            # Calculate metrics
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
            
            # Save some predictions for debugging
            if threshold == 0.5:  # Only for the main threshold
                results['predictions'][threshold] = merged[['Filename_norm', 'Label', 'Speech']].to_dict('records')
        
        # 🔧 FIXED: Log a summary of matches
        if total_matches > 0:
            self.logger.info(f"📈 Total matches found: {total_matches} (across all thresholds)")
        else:
            self.logger.warning(f"⚠️ NO matches were found for any threshold")
        
        return results
    
    def _print_predictions_vs_gt(self, model_name: str, predictions: List[Dict]):
        """!
        @brief Prints a comparison of predictions vs. ground truth for debugging.
        @param model_name The name of the model.
        @param predictions A list of dictionaries, each containing a prediction record.
        """
        if not predictions:
            return
            
        self.logger.info(f"\n🔍 PREDICTIONS vs. GROUND TRUTH - {model_name.upper()}")
        self.logger.info("="*70)
        self.logger.info(f"{'File':<30} {'GT':<5} {'Pred':<5} {'Correct':<10}")
        self.logger.info("-"*70)
        
        correct = 0
        for pred in predictions[:10]:  # Only first 10
            filename = pred['Filename_norm'][:27] + "..." if len(pred['Filename_norm']) > 30 else pred['Filename_norm']
            gt = pred['Label']
            prediction = pred['Speech']
            is_correct = "✅ YES" if gt == prediction else "❌ NO"
            
            if gt == prediction:
                correct += 1
                
            self.logger.info(f"{filename:<30} {gt:<5} {prediction:<5} {is_correct:<10}")
        
        accuracy = correct / len(predictions[:10])
        self.logger.info("-"*70)
        self.logger.info(f"Accuracy on sample: {accuracy:.2%} ({correct}/{len(predictions[:10])})")
        self.logger.info("="*70)
    
    def _save_results(self, model_name: str, results: Dict):
        """!
        @brief Saves the evaluation results to a JSON file.
        @param model_name The name of the model.
        @param results A dictionary with the evaluation results.
        """
        results_file = self.results_dir / f"metrics_{model_name}.json"
        
        # Prepare data for JSON (without detailed predictions)
        json_results = {k: v for k, v in results.items() if k != 'predictions'}
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"💾 Metrics saved to: {results_file}")
    
    def _generate_plots(self, model_name: str, results: Dict):
        """!
        @brief Generates performance plots for a model.
        @param model_name The name of the model.
        @param results A dictionary with the evaluation results.
        """
        if not results.get('thresholds'):
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        thresholds = results['thresholds']
        
        # 1. Precision vs. Threshold
        ax1.plot(thresholds, results['precision'], 'b-o', label='Precision')
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Precision')
        ax1.set_title(f'{model_name.upper()} - Precision vs. Threshold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 2. Recall vs. Threshold
        ax2.plot(thresholds, results['recall'], 'g-o', label='Recall')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Recall')
        ax2.set_title(f'{model_name.upper()} - Recall vs. Threshold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # 3. F1 vs. Threshold
        ax3.plot(thresholds, results['f1'], 'r-o', label='F1-Score')
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('F1-Score')
        ax3.set_title(f'{model_name.upper()} - F1-Score vs. Threshold')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # 4. All metrics together
        ax4.plot(thresholds, results['precision'], 'b-o', label='Precision', alpha=0.7)
        ax4.plot(thresholds, results['recall'], 'g-o', label='Recall', alpha=0.7)
        ax4.plot(thresholds, results['f1'], 'r-o', label='F1-Score', alpha=0.7)
        ax4.plot(thresholds, results['accuracy'], 'm-o', label='Accuracy', alpha=0.7)
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('Score')
        ax4.set_title(f'{model_name.upper()} - All Metrics')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / f"plot_{model_name}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📈 Plot saved to: {plot_file}")
    
    def _generate_comparison_plot(self):
        """! @brief Generates a comparison plot for all evaluated models. """
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
        plt.title('F1-Score Comparison Across Models', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Save comparison plot
        comparison_plot = self.results_dir / "comparison_all_models.png"
        plt.savefig(comparison_plot, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 Comparison plot saved to: {comparison_plot}")
    
    def _print_final_summary(self):
        """! @brief Prints a final summary table comparing all models. """
        if not self.all_results:
            return
            
        self.logger.info(f"\n🎯 FINAL SUMMARY - MODEL COMPARISON")
        self.logger.info("="*80)
        self.logger.info(f"{'Model':<15} {'Best F1':<12} {'@ Threshold':<12} {'Best Precision':<15} {'Best Recall':<12}")
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
        """! @brief Executes the complete evaluation workflow. """
        start_time = time.time()
        
        # Configuration
        max_files = self.config['test_settings']['max_files']
        thresholds = self.config['test_settings']['thresholds']
        
        # Find an available dataset
        dataset_name = None
        audio_files = []
        gt_path = None
        datasets_dict = (self.config.get('datasets') or self.config.get('scenarios'))
        if datasets_dict is None:
            raise KeyError("Configuration must have a 'datasets' or 'scenarios' key at the root level.")
        for name, dataset_config in datasets_dict.items():
            dataset_path = self.base_path / dataset_config['chunks_path']
            if dataset_path.exists():
                audio_files = self._get_audio_files_limited(dataset_config, max_files)
                gt_path = dataset_config['ground_truth']
                dataset_name = name
                break
        
        if not audio_files:
            self.logger.error("❌ No audio files found")
            return
        
        # Load ground truth
        if not os.path.exists(gt_path):
            self.logger.error(f"❌ Ground truth not found: {gt_path}")
            return
            
        gt_df = self._load_ground_truth(gt_path)
        self.logger.info(f"📋 Ground truth loaded: {len(gt_df)} files")
        
        # Evaluate each model
        for model_name, model_config in self.config['models'].items():
            if not model_config.get('enabled', True):
                self.logger.info(f"⏭️  Model {model_name} is disabled, skipping...")
                continue
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🤖 PROCESSING MODEL: {model_name.upper()}")
            self.logger.info(f"{'='*60}")
            
            # 1. Check if previous inference exists
            if self._masks_exist(model_name):
                self.logger.info(f"✅ Masks found for {model_name}, skipping inference")
            else:
                # 2. Load wrapper and run inference
                wrapper = self._load_wrapper(model_config['wrapper'])
                if wrapper is None:
                    self.logger.error(f"❌ Could not load wrapper for {model_name}")
                    continue
                
                self.logger.info(f"✅ Wrapper loaded: {wrapper}")
                
                success = self._run_inference(model_name, wrapper, audio_files, thresholds)
                if not success:
                    self.logger.error(f"❌ Inference failed for {model_name}")
                    continue
            
            # 3. Evaluate against ground truth
            results = self._evaluate_model(model_name, gt_df)
            if not results:
                continue
            
            # 4. Show predictions vs. GT for debugging
            if 0.5 in results.get('predictions', {}):
                self._print_predictions_vs_gt(model_name, results['predictions'][0.5])
            
            # 5. Save results
            self._save_results(model_name, results)
            
            # 6. Generate plots
            self._generate_plots(model_name, results)
            
            # Save for the final summary
            self.all_results[model_name] = results
            
            # Log main metrics
            if results.get('f1'):
                best_f1_idx = np.argmax(results['f1'])
                best_f1 = results['f1'][best_f1_idx]
                best_thresh = results['thresholds'][best_f1_idx]
                self.logger.info(f"🎯 Best F1: {best_f1:.3f} @ threshold {best_thresh:.2f}")
        
        # 7. Comparison plot and final summary
        self._generate_comparison_plot()
        self._print_final_summary()
        
        total_time = time.time() - start_time
        self.logger.info(f"\n✅ Evaluation completed in {total_time:.1f} seconds")
        self.logger.info(f"📁 Results saved in: {self.results_dir}")


def main():
    """!
    @brief Main function to parse arguments and run the VAD evaluation.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete evaluation of VAD models')
    parser.add_argument('--config', default='config_test.yaml', 
                       help='YAML configuration file')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable detailed verbose output')
    parser.add_argument('--mode', choices=['prototype', 'full'], default='prototype',
                       help='Evaluation mode (prototype: few files, full: all files)')
    
    args = parser.parse_args()
    
    # Create evaluator and run
    evaluator = VADEvaluator(args.config, verbose=args.verbose)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()