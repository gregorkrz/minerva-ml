#!/usr/bin/env python3
"""
Compute and store different neutrino energy baselines for MINERvA dataset.

This script processes ROOT files from different playlists and computes various
neutrino energy reconstruction methods:
1. CCQE formula (from reconstructed muon kinematics)
2. Enu_from_muon (from MasterAnaDev_enu_muon)
3. Enu_from_muon+proton (with proton correction)
4. E_mu+E_recoil
5. E_mu+E_recoil_CCinc
6. Reco muon passing the criteria theta<20 degrees and 1.5 < |p| < 20 GeV?
7. Ground truth q0, total lab-frame energy transfer
8. Ground truth q3, three momentum transfer

Results are stored as numpy arrays for each playlist.

"""

import numpy as np
import argparse
import os
import uproot
import pickle
from pathlib import Path
from tqdm import tqdm
from src.dataset.preprocessing import get_dense

mc_part_keys = ["mc_FSPartPx", "mc_FSPartPy", "mc_FSPartPz", "mc_FSPartE", "mc_FSPartPDG"]


def get_muon_kinematics(master_ana_dev):
    """
    Extract muon kinematics from the ROOT file.
    
    Returns:
        dict with keys: 'E', 'px', 'py', 'pz', 'theta', 'phi'
    """
    muon_data = master_ana_dev["muon_corrected_p"].array().to_numpy()

    muon_E = muon_data[:, 3]
    muon_px = muon_data[:, 0]
    muon_py = muon_data[:, 1]
    muon_pz = muon_data[:, 2]
    #muon_theta_cos = muon_pz / np.sqrt(muon_px**2 + muon_py**2 + muon_pz**2)
    
    return {
        'E': muon_E,
        'px': muon_px,
        'py': muon_py,
        'pz': muon_pz,
        #"theta": muon_theta_cos,
    }

def get_muon_filter_CC_paper(muon_kinematics):
    Emu = muon_kinematics['E']
    px = muon_kinematics['px']
    py = muon_kinematics['py']
    pz = muon_kinematics['pz']
    pmu = np.sqrt(px**2 + py**2 + pz**2)

    muon_theta_cos = pz / pmu
    
    # Constants (in MeV)
    proton_mass = 938.2720813505859
    muon_mass = 105.6583745
    
    is_passing_filter = (muon_theta_cos > np.cos(20 * np.pi / 180)) & \
                (pmu > 1.5 * 1000) & \
                (pmu < 20 * 1000)
    return is_passing_filter

def get_q0(muon_kinematics, mc_incomingE):
    return mc_incomingE - muon_kinematics['E']

def get_q3(muon_kinematics, mc_incomingE, q0):
    muon_mass = 105.6583745
    pmu = np.sqrt(muon_kinematics['px']**2 + muon_kinematics['py']**2 + muon_kinematics['pz']**2)
    Qsquared = 2 * mc_incomingE * (muon_kinematics['E'] - pmu* muon_kinematics['pz'] / pmu) - muon_mass**2
    return np.sqrt(Qsquared + q0**2)


def compute_ccqe_formula(muon_kinematics):
    """
    Compute neutrino energy using CCQE formula from reconstructed muon.
    
    E_nu = (2 * M_p * E_mu - m_mu^2) / (2 * (M_p - E_mu + p_mu * cos(theta)))
    
    Args:
        muon_kinematics: dict with muon kinematic variables
    
    Returns:
        E_nu_from_formula: array of neutrino energies (MeV), -1 for invalid events
    """
    Emu = muon_kinematics['E']
    px = muon_kinematics['px']
    py = muon_kinematics['py']
    pz = muon_kinematics['pz']
    
    # Calculate momentum magnitude
    pmu = np.sqrt(px**2 + py**2 + pz**2)
    
    # Calculate cos(theta) from momentum components
    mu_theta_cos = pz / pmu
    
    # Constants (in MeV)
    proton_mass = 938.2720813505859
    muon_mass = 105.6583745
    
    # CCQE formula
    E_nu_formula = (2 * proton_mass * Emu - muon_mass * muon_mass) / \
                   (2 * (proton_mass - Emu + pmu * mu_theta_cos))
    
    # Apply quality cuts: theta < 20 degrees and 1.5 < |p| < 20 GeV
    mask_muon = (mu_theta_cos > np.cos(20 * np.pi / 180)) & \
                (pmu > 1.5 * 1000) & \
                (pmu < 20 * 1000)
    
    # Set invalid events to -1
    E_nu_formula[~mask_muon] = -1
    
    # Also check for invalid muon energy
    E_nu_formula[Emu <= 0] = -1
    
    return E_nu_formula

def get_pion_kinematics(master_ana_dev):
    # Get pion kinematics if there is exactly one charged pion (or one neutral pion with no charged pions).
    # Returns an (N_events, 4) array of four-vectors; rows are zero for events
    # that are not CC or don't have a unique single-pion final state.
    mc_part = get_dense(mc_part_keys, master_ana_dev)
    current = master_ana_dev["mc_current"].array().to_numpy()
    cc_events = np.where(current == 1)[0]
    n_events = len(current)
    pion_four_vectors = np.zeros((n_events, 4))
    for i in range(len(cc_events)):
        ev = cc_events[i]
        event_PDG = mc_part.data[mc_part.bounds[ev]:mc_part.bounds[ev+1]][:, 4].astype(int)
        pion_idx = -1
        n_piplus = np.sum(event_PDG == 211)
        n_piminus = np.sum(event_PDG == -211)
        n_pi0 = np.sum(event_PDG == 111)
        if n_piplus == 1 and n_piminus == 0:
            pion_idx = np.where(event_PDG == 211)[0][0]
        elif n_piplus == 0 and n_piminus == 1:
            pion_idx = np.where(event_PDG == -211)[0][0]
        elif n_piplus == 0 and n_piminus == 0 and n_pi0 == 1:
            pion_idx = np.where(event_PDG == 111)[0][0]
        if pion_idx != -1:
            pion_four_vectors[ev, :] = mc_part.data[mc_part.bounds[ev]:mc_part.bounds[ev+1]][pion_idx, :4]
    return pion_four_vectors

def compute_enu_baselines(root_file_path):
    """
    Compute all neutrino energy baselines for a single ROOT file.
    
    Args:
        root_file_path: path to ROOT file
    
    Returns:
        dict with keys:
            - 'CCQE_formula': E_nu from CCQE formula
            - 'Enu_from_muon': E_nu from MasterAnaDev_enu_muon
            - 'Enu_from_muon+proton': E_nu with proton correction
            - 'E_mu+E_recoil': E_muon + E_recoil
            - 'E_mu+E_recoil_CCinc': E_muon + E_recoil_CCinc
            - 'E_muon': Muon energy (for reference)
    """
    with uproot.open(root_file_path) as uf:
        master_ana_dev = uf["MasterAnaDev"]
        
        # Get muon kinematics
        muon_kinematics = get_muon_kinematics(master_ana_dev)
        E_muon = muon_kinematics['E']
        E_true = master_ana_dev["mc_incomingE"].array().to_numpy()
        
        # 1. CCQE formula
        E_nu_from_formula = compute_ccqe_formula(muon_kinematics)
        
        # 2. Enu from muon (from dataframe)
        E_nu_from_df = master_ana_dev["MasterAnaDev_enu_muon"].array().to_numpy()
        invalid_idx = E_nu_from_df < 0
        E_nu_from_df[invalid_idx] = -1
        
        # 3. Enu from muon + proton correction
        E_nu_from_p_correction = master_ana_dev["MasterAnaDev_enu_proton"].array().to_numpy()
        E_nu_from_p_correction[E_nu_from_p_correction < 0] = 0
        E_nu_from_df_with_p_correction = E_nu_from_df + E_nu_from_p_correction
        E_nu_from_df_with_p_correction[invalid_idx] = -1
        
        # 4. E_mu + E_recoil
        E_recoil = master_ana_dev['MasterAnaDev_hadron_recoil'].array().to_numpy()
        E_recoil = np.where(E_recoil >= 0, E_recoil, -1)
        
        # 5. E_mu + E_recoil_CCinc
        E_recoil_CCinc = master_ana_dev['MasterAnaDev_hadron_recoil_CCInc'].array().to_numpy()
        E_recoil_CCinc = np.where(E_recoil_CCinc >= 0, E_recoil_CCinc, -1)
        
        # Compute combined energies
        invalid_muon = E_muon <= 0
        invalid_E_recoil = E_recoil == -1
        invalid_E_recoil_CCinc = E_recoil_CCinc == -1
        
        E_mu_plus_recoil = E_muon + E_recoil
        E_mu_plus_recoil[invalid_muon | invalid_E_recoil] = -1
        
        E_mu_plus_recoil_CCinc = E_muon + E_recoil_CCinc
        E_mu_plus_recoil_CCinc[invalid_muon | invalid_E_recoil_CCinc] = -1

        # 6. Reco muon passing the criteria theta<20 degrees and 1.5 < |p| < 20 GeV?
        muon_filter_CC_paper = get_muon_filter_CC_paper(muon_kinematics)

        # 7. Ground truth q0, total lab-frame energy transfer
        q0 = get_q0(muon_kinematics, E_true)
        q3 = get_q3(muon_kinematics, E_true, q0)

        # 8. Pion four-vectors (px, py, pz, E) for single-pion CC events
        pion_four_vectors = get_pion_kinematics(master_ana_dev)

        return {
            'CCQE_formula': E_nu_from_formula,
            'Enu_from_muon': E_nu_from_df,
            'Enu_from_muon+proton': E_nu_from_df_with_p_correction,
            'E_mu+E_recoil': E_mu_plus_recoil,
            'E_mu+E_recoil_CCinc': E_mu_plus_recoil_CCinc,
            'E_muon': E_muon,
            'E_true': E_true,
            "E_recoil_only": E_recoil,
            "E_recoil_CCinc_only": E_recoil_CCinc,
            "muon_filter_CC_paper": muon_filter_CC_paper,
            "q0": q0,
            "q3": q3,
            "pion_four_vectors": pion_four_vectors,
        }


def process_playlist(playlist_path, playlist_name):
    """
    Process all ROOT files in a playlist and compute energy baselines.
    
    Args:
        playlist_path: path to directory containing ROOT files
        playlist_name: name of the playlist (e.g., "1A")
    
    Returns:
        dict with concatenated arrays for all baselines
    """
    root_files = sorted([f for f in os.listdir(playlist_path) if f.endswith(".root")])
    
    if len(root_files) == 0:
        print(f"Warning: No ROOT files found in {playlist_path}")
        return None
    
    print(f"[{playlist_name}] Processing {len(root_files)} files...")
    
    # Initialize lists to store results from all files
    all_results = {
        'CCQE_formula': [],
        'Enu_from_muon': [],
        'Enu_from_muon+proton': [],
        'E_mu+E_recoil': [],
        'E_mu+E_recoil_CCinc': [],
        'E_muon': [],
        "E_true": [],
        "E_recoil_only": [],
        "E_recoil_CCinc_only": [],
        "muon_filter_CC_paper": [],
        "q0": [],
        "q3": [],
        "pion_four_vectors": [],
    }
    
    # Process each file
    for root_file in tqdm(root_files, desc=f"[{playlist_name}]"):
        try:
            root_file_path = os.path.join(playlist_path, root_file)
            results = compute_enu_baselines(root_file_path)
            
            # Append results
            for key in all_results.keys():
                all_results[key].append(results[key])
                
        except Exception as e:
            print(f"[{playlist_name}] Error processing {root_file}: {e}")
            continue
    
    # Concatenate all results
    concatenated_results = {}
    for key in all_results.keys():
        if len(all_results[key]) > 0:
            concatenated_results[key] = np.concatenate(all_results[key])
        else:
            concatenated_results[key] = np.array([])
    
    # Print statistics
    n_events = len(concatenated_results['CCQE_formula'])
    print(f"\n[{playlist_name}] Statistics:")
    print(f"  Total events: {n_events}")
    for key in ['CCQE_formula', 'Enu_from_muon', 'Enu_from_muon+proton', 
                'E_mu+E_recoil', 'E_mu+E_recoil_CCinc', 'E_true', 'muon_filter_CC_paper', 'q0', 'q3']:
        valid_events = (concatenated_results[key] > 0).sum()
        print(f"  {key}: {valid_events}/{n_events} valid ({100*valid_events/n_events:.1f}%)")
    
    return concatenated_results


def main():
    parser = argparse.ArgumentParser(
        description="Compute neutrino energy baselines for MINERvA dataset"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Base directory containing playlist subdirectories (e.g., /scratch/MINERvA/raw_data/MediumEnergy_FHC_StandardMC_Playlist/)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the computed baselines"
    )
    parser.add_argument(
        "--playlist",
        type=str,
        default=None,
        help="Process only a specific playlist (e.g., '1A'). If not specified, all playlists will be processed."
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of playlists
    if args.playlist:
        playlists = [args.playlist]
    else:
        playlists = sorted([d for d in os.listdir(args.input_dir) 
                          if os.path.isdir(os.path.join(args.input_dir, d))])
    
    print(f"Processing playlists: {playlists}")
    print("=" * 80)
    
    # Process each playlist
    all_playlist_results = {}
    
    for playlist in playlists:
        playlist_path = os.path.join(args.input_dir, playlist)
        
        if not os.path.exists(playlist_path):
            print(f"Warning: Playlist directory {playlist_path} does not exist, skipping...")
            continue
        
        results = process_playlist(playlist_path, playlist)
        
        if results is not None:
            all_playlist_results[playlist] = results
            # Save results for this playlist
            #output_file = os.path.join(args.output_dir, f"{playlist}_enu_baselines.pkl")
            #with open(output_file, 'wb') as f:
            #    pickle.dump(results, f)
            # Also save as numpy arrays for easier loading
            output_npz = os.path.join(args.output_dir, f"{playlist}_enu_baselines.npz")
            np.savez(output_npz, **results)
            print(f"✓ [{playlist}] Saved to {output_npz}")
        print("=" * 80)
    
    # Save combined results
    #combined_output = os.path.join(args.output_dir, "all_playlists_enu_baselines.pkl")
    #with open(combined_output, 'wb') as f:
    #    pickle.dump(all_playlist_results, f)
    
    print(f"\n✓ All results saved to {args.output_dir}")
    #print(f"✓ Combined results saved to {combined_output}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for playlist, results in all_playlist_results.items():
        n_events = len(results['CCQE_formula'])
        print(f"{playlist}: {n_events} events")


if __name__ == "__main__":
    main()
