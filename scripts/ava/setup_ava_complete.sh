#!/usr/bin/env bash
set -euo pipefail

# AVA-Speech Complete Setup and Fix Script
# Addresses the parameter name error and path misalignment issues

echo "=== AVA-Speech Complete Setup & Fix ==="

# --- Path Configuration ---
META=datasets/ava/metadata
RAW=datasets/ava/raw
WAV=datasets/ava/wav16k
CHUNKS=datasets/ava/chunks_ALL
GT_DIR=ground_truth/ava  # Standard location for GT files
mkdir -p "$META" "$RAW" "$WAV" "$CHUNKS" "$GT_DIR"

echo "Paths configured:"
echo "  Metadata: $META"
echo "  Raw videos: $RAW"
echo "  16kHz WAV: $WAV"
echo "  Chunks: $CHUNKS"
echo "  Ground truth: $GT_DIR"

# --- Step 1: Get Official File List ---
echo -e "\n1. Downloading official AVA-Speech file list..."
curl -fsSL "https://s3.amazonaws.com/ava-dataset/annotations/ava_speech_file_names_v1.txt" \
  | tr ' ' '\n' | sed '/^$/d' > "$META/ava_speech_file_names_v1.txt"
echo "   File list saved: $META/ava_speech_file_names_v1.txt"

# --- Step 2: Download Videos from S3 ---
echo -e "\n2. Downloading videos from S3..."
N=15  # Number of videos to download
PAR=4 # Parallel downloads

tail -n +6 "$META/ava_speech_file_names_v1.txt" | head -n $N | \
  xargs -P$PAR -I{} bash -c '
    FN="{}"
    URL="https://s3.amazonaws.com/ava-dataset/trainval/${FN}"
    OUT="'"$RAW"'/${FN}"
    if [ ! -s "$OUT" ]; then
      echo "[↓] $FN"
      curl -fSL "$URL" -o "$OUT"
    else
      echo "[✓] $FN (already exists)"
    fi
  '

# --- Step 3: Convert to 16kHz WAV (15:00-30:00 segment) ---
echo -e "\n3. Converting to 16kHz WAV (15:00-30:00 segments)..."
find "$RAW" -type f \( -name "*.mp4" -o -name "*.mkv" -o -name "*.webm" \) -print0 | \
  xargs -0 -I{} bash -c '
    IN="{}"; VID="$(basename "$IN")"; VID="${VID%.*}"
    OUT="'"$WAV"'/${VID}.wav"
    if [ ! -s "$OUT" ]; then
      echo "[ffmpeg] $VID (15:00→30:00)"
      ffmpeg -y -ss 900 -t 900 -i "$IN" -ac 1 -ar 16000 "$OUT" < /dev/null 2>/dev/null
    else
      echo "[✓] $VID.wav (already exists)"
    fi
  '

# Generate list of successfully processed videos
ls "$WAV"/*.wav 2>/dev/null | sed -E 's#.*/([^/]+)\.wav#\1#' > "$META/available_videos.txt" || touch "$META/available_videos.txt"
AVAILABLE_COUNT=$(wc -l < "$META/available_videos.txt")
echo "   Successfully processed: $AVAILABLE_COUNT videos"

# --- Step 4: Generate Chunks and Ground Truth (FIXED PARAMETERS) ---
echo -e "\n4. Generating 4s chunks and ground truth..."

# Clean existing chunks
rm -rf "$CHUNKS"/*

# FIXED: Use correct parameter name --min-positive-sec (not --min-pos-dur-sec)
python scripts/ava/prepare_ava_chunks.py \
  --labels "$META/ava_speech_labels_v1.csv" \
  --wav16k-dir "$WAV" \
  --chunks-dir "$CHUNKS" \
  --gt-cmf "$GT_DIR/cmf.csv" \
  --gt-cmfv "$GT_DIR/cmfv.csv" \
  --videos-subset "$META/available_videos.txt" \
  --window-sec 4.0 \
  --hop-sec 4.0 \
  --min-positive-sec 0.5

# Verify results
CHUNK_COUNT=$(ls "$CHUNKS"/*.wav 2>/dev/null | wc -l || echo "0")
CMF_POSITIVE=$(tail -n +2 "$GT_DIR/cmf.csv" 2>/dev/null | grep -c "True" || echo "0")
CMFV_POSITIVE=$(tail -n +2 "$GT_DIR/cmfv.csv" 2>/dev/null | grep -c "True" || echo "0")

echo "   Chunks generated: $CHUNK_COUNT"
echo "   CMF positive chunks: $CMF_POSITIVE"
echo "   CMFV positive chunks: $CMFV_POSITIVE"

# --- Step 5: Verify GT-Chunk Alignment ---
echo -e "\n5. Verifying GT-Chunk alignment..."
if [ -f "$GT_DIR/cmf.csv" ] && [ $CHUNK_COUNT -gt 0 ]; then
  # Extract chunk names from GT and compare with actual files
  awk -F, 'NR>1{print $1".16kHz.wav"}' "$GT_DIR/cmf.csv" | sort > /tmp/gt_expected.list
  (cd "$CHUNKS" && ls *.wav | sort > /tmp/chunks_actual.list)
  
  MISSING_COUNT=$(comm -23 /tmp/gt_expected.list /tmp/chunks_actual.list | wc -l)
  EXTRA_COUNT=$(comm -13 /tmp/gt_expected.list /tmp/chunks_actual.list | wc -l)
  
  echo "   Missing chunks (in GT but not in chunks/): $MISSING_COUNT"
  echo "   Extra chunks (in chunks/ but not in GT): $EXTRA_COUNT"
  
  if [ $MISSING_COUNT -eq 0 ] && [ $EXTRA_COUNT -eq 0 ]; then
    echo "   ✅ Perfect alignment between GT and chunks!"
  else
    echo "   ⚠️ Alignment issues detected"
    if [ $MISSING_COUNT -gt 0 ]; then
      echo "   First 5 missing:"
      comm -23 /tmp/gt_expected.list /tmp/chunks_actual.list | head -5
    fi
  fi
  
  rm -f /tmp/gt_expected.list /tmp/chunks_actual.list
else
  echo "   ⚠️ Cannot verify alignment (missing GT or chunks)"
fi

# --- Summary ---
echo -e "\n=== SETUP COMPLETE ==="
echo "Videos processed: $AVAILABLE_COUNT"
echo "Chunks generated: $CHUNK_COUNT"
echo "CMF scenarios ready: $CMF_POSITIVE positive chunks"
echo "CMFV scenarios ready: $CMFV_POSITIVE positive chunks"
echo ""
echo "Ground truth files:"
echo "  CMF:  $GT_DIR/cmf.csv"
echo "  CMFV: $GT_DIR/cmfv.csv"
echo "Chunks directory: $CHUNKS"
echo ""

if [ $CHUNK_COUNT -gt 0 ] && [ $CMF_POSITIVE -gt 0 ]; then
  echo "✅ Ready for evaluation! Run:"
  echo "   python scripts/run_evaluation.py --config configs/config_ava_cmf.yaml --verbose"
  echo "   python scripts/run_evaluation.py --config configs/config_ava_cmfv.yaml --verbose"
else
  echo "❌ Setup incomplete. Check error messages above."
  exit 1
fi