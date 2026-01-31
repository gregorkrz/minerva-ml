# Neutrino Energy Baseline Computation Scripts

This directory contains scripts to compute and analyze different neutrino energy reconstruction baselines for the MINERvA dataset.

## Overview

The scripts compute five different neutrino energy reconstruction methods:

1. **CCQE formula**: Computed from reconstructed muon kinematics using the CCQE formula:
   ```
   E_nu = (2 * M_p * E_mu - m_mu^2) / (2 * (M_p - E_mu + p_mu * cos(theta)))
   ```
   - Applies quality cuts: theta < 20°, 1.5 < |p_mu| < 20 GeV

2. **Enu_from_muon**: From `MasterAnaDev_enu_muon` field in ROOT files

3. **Enu_from_muon+proton**: Enu_from_muon with proton correction from `MasterAnaDev_enu_proton`

4. **E_mu+E_recoil**: Sum of muon energy and hadronic recoil energy (`MasterAnaDev_hadron_recoil`)

5. **E_mu+E_recoil_CCinc**: Sum of muon energy and CCInc hadronic recoil energy (`MasterAnaDev_hadron_recoil_CCInc`)

## Scripts

### 1. `compute_enu_baselines.py`

Main script to compute baselines from ROOT files.

**Usage:**

```bash
# Process all playlists
python compute_enu_baselines.py \
    --input-dir /scratch/MINERvA/raw_data/MediumEnergy_FHC_StandardMC_Playlist \
    --output-dir /scratch/MINERvA/enu_baselines

# Process a specific playlist
python compute_enu_baselines.py \
    --input-dir /scratch/MINERvA/raw_data/MediumEnergy_FHC_StandardMC_Playlist \
    --output-dir /scratch/MINERvA/enu_baselines \
    --playlist 1A
```

**Arguments:**
- `--input-dir`: Base directory containing playlist subdirectories (1A, 1B, etc.)
- `--output-dir`: Directory to save computed baselines
- `--playlist`: (Optional) Process only a specific playlist

**Output:**
- `{playlist}_enu_baselines.pkl`: Pickle file with all baselines for the playlist
- `{playlist}_enu_baselines.npz`: NumPy compressed file (easier to load)
- `all_playlists_enu_baselines.pkl`: Combined results from all playlists

**Output Format:**

Each file contains a dictionary with keys:
```python
{
    'CCQE_formula': np.array,           # Shape: (n_events,)
    'Enu_from_muon': np.array,          # Shape: (n_events,)
    'Enu_from_muon+proton': np.array,   # Shape: (n_events,)
    'E_mu+E_recoil': np.array,          # Shape: (n_events,)
    'E_mu+E_recoil_CCinc': np.array,    # Shape: (n_events,)
    'E_muon': np.array,                 # Shape: (n_events,) - for reference
}
```

All energies are in **MeV**. Invalid events are marked with `-1`.

### 2. `analyze_enu_baselines.py`

Script to analyze and visualize computed baselines.

**Usage:**

```bash
# Analyze a single playlist
python analyze_enu_baselines.py \
    --input-file /scratch/MINERvA/enu_baselines/1A_enu_baselines.npz \
    --output-dir ./plots \
    --playlist 1A

# Analyze all playlists
python analyze_enu_baselines.py \
    --input-file /scratch/MINERvA/enu_baselines/all_playlists_enu_baselines.pkl \
    --output-dir ./plots

# Just print statistics (no plots)
python analyze_enu_baselines.py \
    --input-file /scratch/MINERvA/enu_baselines/1A_enu_baselines.npz \
    --no-plots
```

**Arguments:**
- `--input-file`: Path to baseline file (.pkl or .npz)
- `--output-dir`: (Optional) Directory to save plots
- `--playlist`: (Optional) Playlist name for labeling
- `--no-plots`: Skip plotting, only print statistics

**Output:**
- Statistics printed to console
- `{playlist}_distributions.png`: Histograms of each baseline
- `{playlist}_comparisons.png`: 2D comparisons between baselines

### 3. `run_compute_enu_baselines.sh`

Example bash script to run the computation.

## Loading Data in Python

### Load from pickle:
```python
import pickle

with open('1A_enu_baselines.pkl', 'rb') as f:
    baselines = pickle.load(f)

# Access specific baseline
ccqe_energies = baselines['CCQE_formula']
valid_mask = ccqe_energies > 0
valid_energies = ccqe_energies[valid_mask]
```

### Load from npz:
```python
import numpy as np

data = np.load('1A_enu_baselines.npz')
ccqe_energies = data['CCQE_formula']
enu_muon = data['Enu_from_muon']
```

### Load all playlists:
```python
import pickle

with open('all_playlists_enu_baselines.pkl', 'rb') as f:
    all_baselines = pickle.load(f)

# Access specific playlist
playlist_1A = all_baselines['1A']
ccqe_1A = playlist_1A['CCQE_formula']
```

## Example Workflow

```bash
# 1. Compute baselines for all playlists
python compute_enu_baselines.py \
    --input-dir /scratch/MINERvA/raw_data/MediumEnergy_FHC_StandardMC_Playlist \
    --output-dir /scratch/MINERvA/enu_baselines

# 2. Analyze results for a specific playlist
python analyze_enu_baselines.py \
    --input-file /scratch/MINERvA/enu_baselines/1A_enu_baselines.npz \
    --output-dir ./plots/1A \
    --playlist 1A

# 3. Analyze all playlists together
python analyze_enu_baselines.py \
    --input-file /scratch/MINERvA/enu_baselines/all_playlists_enu_baselines.pkl \
    --output-dir ./plots/all
```

## Notes

- **Invalid Events**: Events with invalid reconstructions are marked with `-1`
- **Energy Units**: All energies are stored in **MeV**
- **Quality Cuts**: CCQE formula applies standard MINERvA cuts (theta < 20°, momentum range)
- **Memory**: The script processes files sequentially to manage memory usage
- **Progress**: Uses tqdm for progress bars during processing

## Integration with Training

To use these baselines as regression targets in OmniLearned:

```python
import numpy as np

# Load baselines
baselines = np.load('1A_enu_baselines.npz')

# Choose which baseline to use as target
target_energies = baselines['Enu_from_muon+proton']  # or any other baseline

# Filter valid events
valid_mask = target_energies > 0
valid_energies = target_energies[valid_mask]

# Use as regression target in your dataloader
```

## Troubleshooting

**Issue**: Script crashes with memory error
- **Solution**: Process playlists one at a time using `--playlist` flag

**Issue**: No valid events in output
- **Solution**: Check that ROOT files contain the required branches (MasterAnaDev_muon_E, etc.)

**Issue**: Different number of events than expected
- **Solution**: Some events may be filtered due to quality cuts or invalid reconstructions

## References

- MINERvA CCQE analysis: [arXiv:1305.7513](https://arxiv.org/abs/1305.7513)
- Neutrino energy reconstruction methods: MINERvA Technical Notes
