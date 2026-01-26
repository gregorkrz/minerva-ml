from pathlib import Path
import ROOT
import matplotlib.pyplot as plt
import numpy as np
import os
import uproot
import awkward as ak
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.resolution_tools import find_narrowest_interval
from src.preprocessing import get_event_repr, get_muons, get_photons, get_dense, remove_overflows, get_global_features, get_event_labels
import argparse
import time

DATASETS = {}
#for playlist in ["1A", "1B", "1C", "1D", "1E", "1F", "1G", "1L", "1M", "1N", "1O", "1P"]:
for playlist in ["1A", "1B", "1C", "1D", "1E", "1F", "1G", "1L", "1M", "1N", "1O", "1P"]:
    DATASETS[playlist] = f"/scratch/MINERvA/raw_data/MediumEnergy_FHC_StandardMC_Playlist/{playlist}"
MAX_OBJECTS = 100

mc_part_keys = ["mc_FSPartPx", "mc_FSPartPy", "mc_FSPartPz", "mc_FSPartE", "mc_FSPartPDG"]
prong_keys = ["prong_part_pos", "prong_part_E", "prong_part_score", "prong_part_mass", "prong_part_charge", "prong_part_pid"]
blob_keys = ["MasterAnaDev_BlobX", "MasterAnaDev_BlobY", "MasterAnaDev_BlobZ", "MasterAnaDev_BlobT", "MasterAnaDev_BlobTPos", "MasterAnaDev_BlobTotalE"]


def process_playlist(playlist, dataset_path, output_dir="/data"):
    """
    Process all ROOT files in a playlist sequentially (appending to the same HDF5 file).
    
    Args:
        playlist: Playlist name (e.g., "1A")
        dataset_path: Path to directory containing ROOT files
        output_dir: Output directory for HDF5 files
    
    Returns:
        Tuple of (playlist, num_files_processed, num_files_failed, errors)
    """
    output_file = f"{output_dir}/events_{playlist}.h5"
    root_files = sorted([f for f in os.listdir(dataset_path) if f.endswith(".root")])
    
    print(f"[{playlist}] Starting processing of {len(root_files)} files")
    start_time = time.time()
    num_processed = 0
    num_failed = 0
    errors = []
    root_file_names = [] # List of root file names in order in which they were processed
    event_boundaries = [0] # List of event boundaries in the HDF5 file
    total_events_written = 0
    for i, root_file in enumerate(root_files):
        try:
            root_file_path = os.path.join(dataset_path, root_file)
            with uproot.open(root_file_path) as f:
                master_ana_dev = f["MasterAnaDev"]
                muons = get_muons(master_ana_dev)
                photons = get_photons(master_ana_dev)
                blobs = get_dense(blob_keys, master_ana_dev)
                prongs = get_dense(prong_keys, master_ana_dev)
                muons = remove_overflows(muons)
                global_features = get_global_features(master_ana_dev)
                truth_labels = get_event_labels(master_ana_dev)
                n_events_written = get_event_repr(
                    muons, photons, blobs, prongs, 
                    global_features, truth_labels, 
                    max_objects=MAX_OBJECTS, 
                    output_file=output_file
                )
                total_events_written += n_events_written
                event_boundaries.append(total_events_written)
                root_file_names.append(root_file)
            num_processed += 1
            if (i + 1) % 10 == 0:  # Progress update every 10 files
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (len(root_files) - i - 1)
                print(f"[{playlist}] Progress: {i+1}/{len(root_files)} files ({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)")
            
        except Exception as e:
            num_failed += 1
            error_msg = f"Error in {root_file}: {str(e)}"
            errors.append(error_msg)
            print(f"[{playlist}] ✗ {error_msg}")
    
    elapsed = time.time() - start_time
    print(f"[{playlist}] ✓ Complete: {num_processed}/{len(root_files)} files succeeded in {elapsed:.1f}s")
    
    return (playlist, num_processed, num_failed, errors)


def process_all_playlists_parallel(datasets, output_dir="/data", max_workers=None):
    """
    Process multiple playlists in parallel (each playlist processes its files sequentially).
    
    Args:
        datasets: Dictionary of {playlist: path}
        output_dir: Output directory for HDF5 files
        max_workers: Maximum number of playlists to process simultaneously (None = all)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    playlists = list(datasets.keys())
    
    if max_workers is None:
        max_workers = len(playlists)
    
    print("=" * 80)
    print(f"Processing {len(playlists)} playlists in parallel")
    print(f"Max concurrent playlists: {max_workers}")
    print(f"Each playlist will process files sequentially (appending to same HDF5)")
    print("=" * 80)
    
    all_results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all playlists
        future_to_playlist = {
            executor.submit(process_playlist, playlist, datasets[playlist], output_dir): playlist
            for playlist in playlists
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_playlist):
            playlist = future_to_playlist[future]
            try:
                result = future.result()
                all_results[playlist] = result
            except Exception as e:
                print(f"[{playlist}] Playlist processing failed: {e}")
                import traceback
                traceback.print_exc()
                all_results[playlist] = (playlist, 0, 0, [str(e)])
    
    return all_results


def print_summary(all_results):
    """Print a summary of all processing results."""
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    
    total_files = 0
    total_success = 0
    total_failed = 0
    
    for playlist in sorted(all_results.keys()):
        result = all_results[playlist]
        _, num_success, num_failed, errors = result
        
        total_files += (num_success + num_failed)
        total_success += num_success
        total_failed += num_failed
        
        status = "✓" if num_failed == 0 else "✗"
        print(f"{status} {playlist}: {num_success}/{num_success + num_failed} files succeeded")
        
        # Print errors if any
        if errors:
            for error in errors[:3]:  # Show first 3 errors
                print(f"    {error}")
            if len(errors) > 3:
                print(f"    ... and {len(errors) - 3} more errors")
    
    print("=" * 80)
    print(f"TOTAL: {total_success}/{total_files} files succeeded, {total_failed} failed")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MINERvA ROOT files to HDF5 format")
    parser.add_argument("--output-dir", default="/data", help="Output directory for HDF5 files")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Maximum number of playlists to process simultaneously (default: all)")
    parser.add_argument("--playlists", nargs="+", default=None,
                        help="Specific playlists to process (default: all)")
    
    args = parser.parse_args()
    
    # Filter datasets if specific playlists requested
    if args.playlists:
        datasets_to_process = {k: v for k, v in DATASETS.items() if k in args.playlists}
    else:
        datasets_to_process = DATASETS
    
    print(f"Selected playlists: {list(datasets_to_process.keys())}")
    
    start_time = time.time()
    
    all_results = process_all_playlists_parallel(
        datasets_to_process,
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )
    
    total_time = time.time() - start_time
    
    print_summary(all_results)
    print(f"\nTotal processing time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")


