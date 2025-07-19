#!/bin/bash
set -e

DATASET_URL="https://archive.org/download/chime-home/chime_home.tar.gz"
TARGET_DIR="datasets/chime"

echo "🎵 Downloading CHiME-Home Dataset (~3.9GB)..."

wget -O chime_home.tar.gz "$DATASET_URL"
tar -xzf chime_home.tar.gz
mkdir -p "$TARGET_DIR"
cp -r chime_home/chunks "$TARGET_DIR/"
rm -rf chime_home

echo "✅ Dataset ready: $(ls $TARGET_DIR/chunks/*.wav | wc -l) audio files"
