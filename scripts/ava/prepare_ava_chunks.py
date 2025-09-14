#!/usr/bin/env python3
"""
AVA-Speech Chunking and Ground Truth Generation Script

Converts AVA-Speech dataset to CHiME-compatible format:
- Creates 4s non-overlapping chunks from 16kHz WAV files
- Generates boolean ground truth CSVs (Chunk,Condition) format
- Maps AVA labels to CMF/CMFV scenarios as defined

AVA Label Mapping:
- CMF (speech only):  CLEAN_SPEECH, SPEECH_WITH_NOISE → True
                      NO_SPEECH, SPEECH_WITH_MUSIC → False
- CMFV (speech+music): CLEAN_SPEECH, SPEECH_WITH_NOISE, SPEECH_WITH_MUSIC → True
                       NO_SPEECH → False

Compatible with existing VAD evaluation pipeline.
"""

import argparse
import csv
import json
import os
import subprocess
import wave
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# AVA-Speech label mapping constants
AVA_LABELS = {
    'NO_SPEECH', 
    'CLEAN_SPEECH', 
    'SPEECH_WITH_MUSIC', 
    'SPEECH_WITH_NOISE'
}

# Scenario mappings
CMF_POSITIVE = {'CLEAN_SPEECH', 'SPEECH_WITH_NOISE'}
CMF_NEGATIVE = {'NO_SPEECH', 'SPEECH_WITH_MUSIC'}

CMFV_POSITIVE = {'CLEAN_SPEECH', 'SPEECH_WITH_NOISE', 'SPEECH_WITH_MUSIC'}
CMFV_NEGATIVE = {'NO_SPEECH'}

# AVA labels are for 15:00-30:00 region in original video
# We subtract this offset to align with our trimmed WAV files
AVA_TIME_OFFSET_SEC = 900.0  # 15 minutes

def load_ava_labels(labels_csv: str) -> Dict[str, List[Tuple[float, float, str]]]:
    """
    Load AVA-Speech labels and convert to local WAV time.
    
    Args:
        labels_csv: Path to ava_speech_labels_v1.csv
        
    Returns:
        Dict mapping video_id to list of (start_sec, end_sec, label) tuples
        Times are adjusted to 0-based for trimmed WAV files
    """
    labels_by_video = defaultdict(list)
    
    with open(labels_csv, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for row in reader:
            if len(row) != 4:
                continue
                
            video_id, start_str, end_str, label = row
            
            # Convert to local WAV time (subtract 15min offset)
            start_sec = float(start_str) - AVA_TIME_OFFSET_SEC
            end_sec = float(end_str) - AVA_TIME_OFFSET_SEC
            
            # Only keep intervals that intersect [0, 900] (our 15min segment)
            start_clipped = max(0.0, start_sec)
            end_clipped = min(900.0, end_sec)
            
            if end_clipped > 0.0 and start_clipped < 900.0 and end_clipped > start_clipped:
                labels_by_video[video_id].append((start_clipped, end_clipped, label))
    
    # Sort intervals by start time for each video
    for video_id in labels_by_video:
        labels_by_video[video_id].sort(key=lambda x: (x[0], x[1]))
    
    return dict(labels_by_video)

def get_wav_duration(wav_path: str) -> float:
    """Get duration of WAV file in seconds."""
    try:
        with wave.open(wav_path, 'rb') as w:
            frames = w.getnframes()
            sample_rate = w.getframerate()
            return frames / float(sample_rate) if sample_rate > 0 else 0.0
    except Exception:
        return 0.0

def create_chunk_with_ffmpeg(input_wav: str, output_wav: str, start_sec: float, duration_sec: float):
    """Create audio chunk using ffmpeg."""
    cmd = [
        'ffmpeg', '-y',
        '-ss', f'{start_sec:.3f}',
        '-t', f'{duration_sec:.3f}',
        '-i', input_wav,
        '-ac', '1',
        '-ar', '16000',
        output_wav
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {output_wav}")

def generate_chunks(video_id: str, wav_path: str, chunks_dir: str, 
                   window_sec: float = 4.0, hop_sec: float = 4.0) -> List[Tuple[str, float, float, int]]:
    """
    Generate 4s audio chunks from WAV file.
    
    Args:
        video_id: Video ID
        wav_path: Path to 16kHz WAV file
        chunks_dir: Output directory for chunks
        window_sec: Chunk duration in seconds
        hop_sec: Hop size in seconds
        
    Returns:
        List of (chunk_filename, start_sec, end_sec, chunk_idx) tuples
    """
    duration = get_wav_duration(wav_path)
    if duration <= 0:
        return []
    
    # Limit to 15 minutes max (AVA labeled region)
    duration = min(duration, 900.0)
    
    chunks_created = []
    chunk_idx = 0
    
    # Generate non-overlapping windows
    t = 0.0
    while t < duration - 1e-6:  # Small epsilon to avoid floating point issues
        window_end = min(t + window_sec, duration)
        
        # Create chunk filename (compatible with CHiME pattern)
        start_ms = int(round(t * 1000))
        end_ms = int(round(window_end * 1000))
        chunk_name = f"AVA_{video_id}_s{start_ms}_e{end_ms}_chunk{chunk_idx}"
        chunk_filename = f"{chunk_name}.16kHz.wav"
        chunk_path = os.path.join(chunks_dir, chunk_filename)
        
        # Create audio chunk
        try:
            create_chunk_with_ffmpeg(wav_path, chunk_path, t, window_end - t)
            chunks_created.append((chunk_name, t, window_end, chunk_idx))
            print(f"  ✅ Created chunk {chunk_idx}: {t:.1f}s-{window_end:.1f}s")
        except Exception as e:
            print(f"  ❌ Failed to create chunk {chunk_idx}: {e}")
        
        chunk_idx += 1
        t += hop_sec
    
    return chunks_created

def calculate_positive_overlap(intervals: List[Tuple[float, float, str]], 
                              window_start: float, window_end: float,
                              positive_labels: set) -> float:
    """
    Calculate total duration of positive labels overlapping with window.
    
    Args:
        intervals: List of (start, end, label) tuples
        window_start: Window start time
        window_end: Window end time  
        positive_labels: Set of labels considered positive
        
    Returns:
        Total overlap duration in seconds
    """
    total_overlap = 0.0
    
    for start, end, label in intervals:
        if label not in positive_labels:
            continue
            
        # Calculate intersection
        overlap_start = max(start, window_start)
        overlap_end = min(end, window_end)
        
        if overlap_end > overlap_start:
            total_overlap += (overlap_end - overlap_start)
    
    return total_overlap

def generate_ground_truth(chunks_info: List[Tuple[str, str, float, float]], 
                         labels_by_video: Dict[str, List[Tuple[float, float, str]]],
                         positive_labels: set, min_positive_duration: float = 0.5) -> List[Tuple[str, bool]]:
    """
    Generate ground truth labels for chunks.
    
    Args:
        chunks_info: List of (chunk_name, video_id, start_sec, end_sec) tuples
        labels_by_video: AVA labels by video ID
        positive_labels: Set of AVA labels considered positive
        min_positive_duration: Minimum overlap duration to label as positive
        
    Returns:
        List of (chunk_name, is_positive) tuples
    """
    gt_entries = []
    
    for chunk_name, video_id, start_sec, end_sec in chunks_info:
        intervals = labels_by_video.get(video_id, [])
        
        # Calculate positive overlap
        positive_duration = calculate_positive_overlap(
            intervals, start_sec, end_sec, positive_labels
        )
        
        # Determine label based on threshold
        is_positive = positive_duration >= min_positive_duration
        gt_entries.append((chunk_name, is_positive))
    
    return gt_entries

def write_chime_format_csv(gt_entries: List[Tuple[str, bool]], output_path: str):
    """Write ground truth in CHiME format: Chunk,Condition with True/False values."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Chunk', 'Condition'])  # CHiME header format
        
        for chunk_name, is_positive in gt_entries:
            condition = 'True' if is_positive else 'False'
            writer.writerow([chunk_name, condition])

def main():
    parser = argparse.ArgumentParser(
        description='Generate 4s chunks and CHiME-compatible ground truth from AVA-Speech'
    )
    parser.add_argument('--labels', required=True, 
                       help='Path to ava_speech_labels_v1.csv')
    parser.add_argument('--wav16k-dir', required=True,
                       help='Directory containing 16kHz WAV files')
    parser.add_argument('--chunks-dir', required=True,
                       help='Output directory for 4s chunks')
    parser.add_argument('--gt-cmf', required=True,
                       help='Output path for CMF ground truth CSV')
    parser.add_argument('--gt-cmfv', required=True,
                       help='Output path for CMFV ground truth CSV')
    parser.add_argument('--window-sec', type=float, default=4.0,
                       help='Chunk window size in seconds')
    parser.add_argument('--hop-sec', type=float, default=4.0,
                       help='Hop size in seconds (use same as window for non-overlapping)')
    parser.add_argument('--min-positive-sec', type=float, default=0.5,
                       help='Minimum positive overlap to label chunk as positive')
    parser.add_argument('--videos-subset',
                       help='Optional file with video IDs to process (one per line)')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.chunks_dir, exist_ok=True)
    
    print("=== AVA-Speech Chunking and Ground Truth Generation ===")
    print(f"Labels: {args.labels}")
    print(f"WAV directory: {args.wav16k_dir}")
    print(f"Chunks output: {args.chunks_dir}")
    print(f"Window size: {args.window_sec}s, Hop: {args.hop_sec}s")
    print(f"Positive threshold: {args.min_positive_sec}s")
    
    # Load AVA labels
    print("\n📋 Loading AVA-Speech labels...")
    labels_by_video = load_ava_labels(args.labels)
    print(f"Loaded labels for {len(labels_by_video)} videos")
    
    # Determine videos to process
    if args.videos_subset and os.path.exists(args.videos_subset):
        with open(args.videos_subset, 'r') as f:
            target_videos = [line.strip() for line in f if line.strip()]
        print(f"Processing subset of {len(target_videos)} videos")
    else:
        # Process all available WAV files
        wav_files = list(Path(args.wav16k_dir).glob('*.wav'))
        target_videos = [f.stem for f in wav_files]
        print(f"Processing all {len(target_videos)} available videos")
    
    # Process each video
    all_chunks_info = []
    processed_videos = 0
    
    for video_id in target_videos:
        wav_path = os.path.join(args.wav16k_dir, f"{video_id}.wav")
        
        if not os.path.exists(wav_path):
            continue
        
        print(f"\n🎵 Processing video: {video_id}")
        
        # Generate chunks
        chunks_created = generate_chunks(
            video_id, wav_path, args.chunks_dir,
            args.window_sec, args.hop_sec
        )
        
        if not chunks_created:
            print(f"  ⚠️  No chunks created for {video_id}")
            continue
        
        # Add to overall chunk info
        for chunk_name, start_sec, end_sec, _ in chunks_created:
            all_chunks_info.append((chunk_name, video_id, start_sec, end_sec))
        
        processed_videos += 1
        print(f"  📊 Created {len(chunks_created)} chunks")
    
    print(f"\n📊 Processing complete:")
    print(f"  Videos processed: {processed_videos}")
    print(f"  Total chunks created: {len(all_chunks_info)}")
    
    if not all_chunks_info:
        print("❌ No chunks were created. Check input paths and files.")
        return
    
    # Generate CMF ground truth
    print("\n🏷️  Generating CMF ground truth (speech only)...")
    cmf_gt = generate_ground_truth(all_chunks_info, labels_by_video, 
                                  CMF_POSITIVE, args.min_positive_sec)
    cmf_positive = sum(1 for _, is_pos in cmf_gt if is_pos)
    print(f"  CMF: {cmf_positive}/{len(cmf_gt)} positive chunks")
    
    # Generate CMFV ground truth  
    print("🏷️  Generating CMFV ground truth (speech + music)...")
    cmfv_gt = generate_ground_truth(all_chunks_info, labels_by_video,
                                   CMFV_POSITIVE, args.min_positive_sec)
    cmfv_positive = sum(1 for _, is_pos in cmfv_gt if is_pos)
    print(f"  CMFV: {cmfv_positive}/{len(cmfv_gt)} positive chunks")
    
    # Write ground truth files
    print("\n💾 Writing ground truth CSVs...")
    write_chime_format_csv(cmf_gt, args.gt_cmf)
    write_chime_format_csv(cmfv_gt, args.gt_cmfv)
    print(f"  CMF GT: {args.gt_cmf}")
    print(f"  CMFV GT: {args.gt_cmfv}")
    
    # Save processing log
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'stats': {
            'videos_processed': processed_videos,
            'total_chunks': len(all_chunks_info),
            'cmf_positive': cmf_positive,
            'cmfv_positive': cmfv_positive
        },
        'label_mappings': {
            'CMF_POSITIVE': list(CMF_POSITIVE),
            'CMF_NEGATIVE': list(CMF_NEGATIVE),
            'CMFV_POSITIVE': list(CMFV_POSITIVE),
            'CMFV_NEGATIVE': list(CMFV_NEGATIVE)
        }
    }
    
    log_path = os.path.join(os.path.dirname(args.gt_cmf), 'processing_log.json')
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"  Processing log: {log_path}")
    print("\n✅ AVA-Speech chunking and ground truth generation complete!")
    print("Ready for VAD evaluation pipeline.")

if __name__ == '__main__':
    main()

