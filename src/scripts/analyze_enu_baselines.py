#!/usr/bin/env python3
"""
Analyze and visualize the computed neutrino energy baselines.

This script loads the computed baselines and provides statistics and visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from pathlib import Path


def load_baselines(file_path):
    """Load baselines from pickle or npz file."""
    file_path = Path(file_path)
    
    if file_path.suffix == '.pkl':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif file_path.suffix == '.npz':
        data = np.load(file_path)
        return {key: data[key] for key in data.files}
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def print_statistics(baselines, playlist_name=""):
    """Print statistics for the baselines."""
    prefix = f"[{playlist_name}] " if playlist_name else ""
    
    n_events = len(baselines['CCQE_formula'])
    print(f"\n{prefix}Total events: {n_events}")
    print(f"{prefix}{'='*60}")
    
    baseline_names = [
        'CCQE_formula',
        'Enu_from_muon',
        'Enu_from_muon+proton',
        'E_mu+E_recoil',
        'E_mu+E_recoil_CCinc'
    ]
    
    for name in baseline_names:
        if name not in baselines:
            continue
            
        values = baselines[name]
        valid_mask = values > 0
        n_valid = valid_mask.sum()
        
        print(f"\n{prefix}{name}:")
        print(f"  Valid events: {n_valid}/{n_events} ({100*n_valid/n_events:.1f}%)")
        
        if n_valid > 0:
            valid_values = values[valid_mask]
            print(f"  Mean: {valid_values.mean():.2f} MeV ({valid_values.mean()/1000:.2f} GeV)")
            print(f"  Std:  {valid_values.std():.2f} MeV ({valid_values.std()/1000:.2f} GeV)")
            print(f"  Min:  {valid_values.min():.2f} MeV ({valid_values.min()/1000:.2f} GeV)")
            print(f"  Max:  {valid_values.max():.2f} MeV ({valid_values.max()/1000:.2f} GeV)")
            print(f"  Median: {np.median(valid_values):.2f} MeV ({np.median(valid_values)/1000:.2f} GeV)")


def plot_distributions(baselines, output_dir=None, playlist_name=""):
    """Plot distributions of the different baselines."""
    baseline_names = [
        'CCQE_formula',
        'Enu_from_muon',
        'Enu_from_muon+proton',
        'E_mu+E_recoil',
        'E_mu+E_recoil_CCinc'
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, name in enumerate(baseline_names):
        if name not in baselines:
            continue
            
        ax = axes[idx]
        values = baselines[name]
        valid_values = values[values > 0] / 1000  # Convert to GeV
        
        ax.hist(valid_values, bins=100, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Energy (GeV)', fontsize=12)
        ax.set_ylabel('Events', fontsize=12)
        ax.set_title(f'{name}\n({len(valid_values)} valid events)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Mean: {valid_values.mean():.2f} GeV\n'
        stats_text += f'Std: {valid_values.std():.2f} GeV'
        ax.text(0.65, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Remove extra subplot
    fig.delaxes(axes[-1])
    
    title = f'Neutrino Energy Baselines - {playlist_name}' if playlist_name else 'Neutrino Energy Baselines'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / f'{playlist_name}_distributions.png' if playlist_name else Path(output_dir) / 'distributions.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def plot_comparisons(baselines, output_dir=None, playlist_name=""):
    """Plot comparisons between different baseline methods."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    comparisons = [
        ('CCQE_formula', 'Enu_from_muon'),
        ('Enu_from_muon', 'Enu_from_muon+proton'),
        ('CCQE_formula', 'E_mu+E_recoil'),
        ('Enu_from_muon+proton', 'E_mu+E_recoil_CCinc')
    ]
    
    for idx, (name1, name2) in enumerate(comparisons):
        ax = axes.flatten()[idx]
        
        if name1 not in baselines or name2 not in baselines:
            continue
        
        values1 = baselines[name1]
        values2 = baselines[name2]
        
        # Only plot where both are valid
        valid_mask = (values1 > 0) & (values2 > 0)
        v1 = values1[valid_mask] / 1000  # Convert to GeV
        v2 = values2[valid_mask] / 1000
        
        # 2D histogram
        h = ax.hist2d(v1, v2, bins=100, cmap='viridis', cmin=1)
        plt.colorbar(h[3], ax=ax, label='Events')
        
        # Add diagonal line
        min_val = min(v1.min(), v2.min())
        max_val = max(v1.max(), v2.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
        
        ax.set_xlabel(f'{name1} (GeV)', fontsize=11)
        ax.set_ylabel(f'{name2} (GeV)', fontsize=11)
        ax.set_title(f'{name1} vs {name2}\n({len(v1)} events)', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    title = f'Baseline Comparisons - {playlist_name}' if playlist_name else 'Baseline Comparisons'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / f'{playlist_name}_comparisons.png' if playlist_name else Path(output_dir) / 'comparisons.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze computed neutrino energy baselines"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to baseline file (.pkl or .npz)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (if not specified, plots will be displayed)"
    )
    parser.add_argument(
        "--playlist",
        type=str,
        default="",
        help="Playlist name for labeling"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plotting, only print statistics"
    )
    
    args = parser.parse_args()
    
    # Load baselines
    print(f"Loading baselines from {args.input_file}...")
    baselines = load_baselines(args.input_file)
    
    # If it's a dict of playlists, process each
    if isinstance(baselines, dict) and 'CCQE_formula' not in baselines:
        # This is a dict of playlists
        for playlist, data in baselines.items():
            print_statistics(data, playlist)
            
            if not args.no_plots:
                plot_distributions(data, args.output_dir, playlist)
                plot_comparisons(data, args.output_dir, playlist)
    else:
        # Single playlist
        print_statistics(baselines, args.playlist)
        
        if not args.no_plots:
            plot_distributions(baselines, args.output_dir, args.playlist)
            plot_comparisons(baselines, args.output_dir, args.playlist)


if __name__ == "__main__":
    main()
