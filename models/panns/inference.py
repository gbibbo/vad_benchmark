# pytorch/inference.py

import os
import sys
import numpy as np
import librosa
import torch
import pandas as pd

# Importaciones correctas para la estructura de archivos
from .pytorch_utils import move_data_to_device
from . import config
from .models import Cnn14_DecisionLevelAtt


def run_panns_inference(audio_path: str, checkpoint_path: str, model_type: str = 'Cnn14_DecisionLevelAtt') -> dict:
    """
    Performs both Sound Event Detection (framewise) and Audio Tagging (clipwise)
    on an audio file.

    Args:
        audio_path (str): Path to the input audio file.
        checkpoint_path (str): Path to the model checkpoint.
        model_type (str): The model architecture to use.

    Returns:
        dict: A dictionary containing 'sed' (a list of framewise predictions)
              and 'at' (a list of clipwise predictions).
    """
    sample_rate = 32000
    window_size = 1024
    hop_size = 320
    mel_bins = 64
    fmin = 50
    fmax = 14000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes_num = config.classes_num
    labels = config.labels

    # Cargar Modelo
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size,
                  hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax,
                  classes_num=classes_num)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)

    if 'cuda' in str(device):
        model.to(device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
    
    # Cargar Audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    waveform = waveform[None, :]
    waveform = move_data_to_device(waveform, device)

    # Inferencia (una sola vez)
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)

    # --- 1. PROCESAR SOUND EVENT DETECTION (por frame) ---
    framewise_output = batch_output_dict.get('framewise_output')
    sed_results = []
    
    # MODIFICACIÓN: Usamos .nelement() para comprobar si el tensor no está vacío
    if framewise_output is not None and framewise_output.nelement() > 0:
        framewise_output = framewise_output.data.cpu().numpy()[0]
        top_k_sed = 7
        for frame_idx, frame_predictions in enumerate(framewise_output):
            frame_dict = {
                'frame_index': frame_idx,
                'time': round(frame_idx * hop_size / sample_rate, 4),
                'predictions': []
            }
            top_indices = np.argsort(frame_predictions)[-top_k_sed:][::-1]
            for class_idx in top_indices:
                frame_dict['predictions'].append({
                    'class': labels[class_idx],
                    'prob': round(float(frame_predictions[class_idx]), 4)
                })
            sed_results.append(frame_dict)

    # --- 2. PROCESAR AUDIO TAGGING (resumen del clip) ---
    clipwise_output = batch_output_dict.get('clipwise_output')
    at_results = []

    # MODIFICACIÓN: Usamos .nelement() para comprobar si el tensor no está vacío
    if clipwise_output is not None and clipwise_output.nelement() > 0:
        clipwise_output = clipwise_output.data.cpu().numpy()[0]
        top_k_at = 10
        sorted_indexes = np.argsort(clipwise_output)[::-1]
        for i in range(top_k_at):
            at_results.append({
                'class': labels[sorted_indexes[i]],
                'prob': round(float(clipwise_output[sorted_indexes[i]]), 4)
            })

    # --- Devolver ambos resultados ---
    return {
        "sed": sed_results,
        "at": at_results
    }