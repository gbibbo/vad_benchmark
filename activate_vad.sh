#!/usr/bin/env bash
conda activate vad_benchmark_py310
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
echo "VAD-Benchmark environment activated: $(python -V)"
