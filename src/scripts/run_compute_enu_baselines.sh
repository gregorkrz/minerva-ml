#!/bin/bash
# Example script to compute neutrino energy baselines for MINERvA dataset

# Set paths
INPUT_DIR="/scratch/MINERvA/raw_data/MediumEnergy_FHC_StandardMC_Playlist"
OUTPUT_DIR="/data/Minerva/20260127_nested_split/enu_baselines"

# Process all playlists
python -m src.scripts.compute_enu_baselines --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR" --playlist 1A

# Or process a specific playlist:
# python compute_enu_baselines.py \
#     --input-dir "$INPUT_DIR" \
#     --output-dir "$OUTPUT_DIR" \
#     --playlist 1A

