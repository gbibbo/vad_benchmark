# src/evaluation/calculate_effectiveness.py
# src/evaluation/calculate_effectiveness.py  – debug v3-fix2
"""
Calculator con depuración.
Se corrige acceso a args.ground_truth (antes había un guion)
y se renombra correctamente la opción --plot.
"""

from __future__ import annotations
import argparse, re, json
from pathlib import Path
from typing import Dict, List
import os, re  
import pandas as pd
from sklearn.metrics import (precision_recall_fscore_support,
                             classification_report, auc)

# ---------- normalización ---------------------------------------------------
_PAT_S = re.compile(r"\.s\d+")

def norm(stem: str) -> str:
    """
    Devuelve solo el basename sin extensión ni sufijos '.16kHz', '.48kHz', etc.
    Así 'speech/speech-librivox-0000_000.16kHz.wav' → 'speech-librivox-0000_000'
    """
    stem = os.path.basename(stem.replace("\\", "/"))   # ← NUEVO: quita carpeta
    stem = (stem.replace(".48kHz.wav", "")
                 .replace(".16kHz.wav", "")
                 .replace(".48kHz", "")
                 .replace(".16kHz", "")
                 .replace(".wav", ""))
    return _PAT_S.sub("", stem)

# ---------- utilidades ------------------------------------------------------
def load_ground_truth(csv_path: Path, dbg=False) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if {"Chunk", "Condition"} - set(df.columns):
        raise ValueError("CSV must tener columnas Chunk y Condition")
    df["Chunk_clean"] = df["Chunk"].apply(norm)
    if dbg:
        print(f"[DBG] GT rows:{len(df)} unique:{df['Chunk_clean'].nunique()}")
        print("[DBG] GT sample:", df["Chunk_clean"].iloc[0])
    return df[["Chunk_clean", "Condition"]]

def enumerate_masks(th_dir: Path) -> Dict[float, Path]:
    out: Dict[float, Path] = {}
    for f in th_dir.glob("mask_*.csv"):
        m = re.match(r"mask_([-+]?\d*\.\d+|\d+)\.csv", f.name)
        if m: out[float(m.group(1))] = f
    return dict(sorted(out.items()))

# ---------- evaluación ------------------------------------------------------
def evaluate_folder(th_dir: Path, gt_csv: Path, dbg=False):
    gt = load_ground_truth(gt_csv, dbg)
    masks = enumerate_masks(th_dir)
    if not masks:
        raise RuntimeError("No mask_Θ.csv files")

    thr, prec0, rec0, f10, prec1, rec1, f11, fpr, tpr = ([] for _ in range(9))

    for theta, path in masks.items():
        df_m = pd.read_csv(path)
        df_m["Filename_clean"] = df_m["Filename"].apply(norm)

        if dbg and theta == 0.0:
            print(f"[DBG θ={theta}] mask rows:{len(df_m)}")
            print("[DBG] mask sample:", df_m["Filename_clean"].iloc[0])

        merged = pd.merge(gt, df_m,
                          left_on="Chunk_clean",
                          right_on="Filename_clean", how="inner")

        if dbg and theta == 0.0:
            print(f"[DBG θ={theta}] merged rows:{len(merged)}")

        if merged.empty:
            print(f"[WARN] θ={theta}: merge empty")
            thr.append(theta); prec0.append(0); rec0.append(0); f10.append(0)
            prec1.append(0); rec1.append(0); f11.append(0); fpr.append(0); tpr.append(0)
            continue

        y_true = merged["Condition"].astype(int)
        y_pred = merged["Speech"].astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, zero_division=0, labels=[0, 1])

        thr.append(theta)
        prec0.append(p[0]); rec0.append(r[0]); f10.append(f[0])
        prec1.append(p[1]); rec1.append(r[1]); f11.append(f[1])
        fpr.append(1 - r[0]); tpr.append(r[1])

        if abs(theta - 0.20) < 1e-6:
            print(f"\n=== Classification report @ θ=0.20 ===")
            print(classification_report(
                y_true, y_pred,
                labels=[0,1],
                target_names=["No-speech(0)", "Speech(1)"],
                zero_division=0))

    return dict(thresholds=thr, prec0=prec0, rec0=rec0, f10=f10,
                prec1=prec1, rec1=rec1, f11=f11,
                fpr=fpr, tpr=tpr, auc=auc(fpr, tpr))

# ---------- main ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds-dir", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--plot", action="store_true")   # ←  FIX: sin espacio
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    res = evaluate_folder(Path(args.thresholds_dir),
                          Path(args.ground_truth),   # ←  FIX: argumento correcto
                          dbg=args.debug)

    with open(f"{args.out_prefix}_metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"[OK] JSON saved → {args.out_prefix}_metrics.json")

if __name__ == "__main__":
    main()
