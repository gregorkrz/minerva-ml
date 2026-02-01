from pathlib import Path
import ROOT
import matplotlib.pyplot as plt
import numpy as np
import os
import uproot
import awkward as ak

from src.dataset.preprocessing import get_dense, get_photons, get_muons

DATASETS = {}
for playlist in ["1A", "1B", "1C", "1D", "1E", "1F", "1G", "1L", "1M", "1N", "1O", "1P"]:
    DATASETS[playlist] = f"/scratch/MINERvA/raw_data/MediumEnergy_FHC_StandardMC_Playlist/{playlist}"

mc_part_keys = ["mc_FSPartPx", "mc_FSPartPy", "mc_FSPartPz", "mc_FSPartE", "mc_FSPartPDG"]
prong_keys = ["prong_part_pos", "prong_part_E", "prong_part_score", "prong_part_mass", "prong_part_charge", "prong_part_pid"]
blob_keys = ["MasterAnaDev_BlobX", "MasterAnaDev_BlobY", "MasterAnaDev_BlobZ", "MasterAnaDev_BlobT", "MasterAnaDev_BlobTPos", "MasterAnaDev_BlobTotalE"]

def get_histograms(master_ana_dev):
    bins_N_muons = np.array([0, 1, 2])
    bins_N_mc_part = np.arange(0, 150, 1)
    bins_N_prong = np.arange(0, 15, 1)
    bins_N_blob = np.arange(0, 150, 1)
    bins_N_photons = np.array([0, 1, 2])
    bins_N_all = np.arange(0, 150, 1)
    mc_part = get_dense(mc_part_keys, master_ana_dev)
    prong = get_dense(prong_keys, master_ana_dev)
    blob = get_dense(blob_keys, master_ana_dev)
    photons = get_photons(master_ana_dev)
    muons = get_muons(master_ana_dev)
    # create a numpy histogram of mc_part.n, prong.n, blob.n, photons.n, muons.n, all=sum of all except mc_part
    N_mc_part = mc_part.n
    N_prong = prong.n
    N_blob = blob.n
    N_photons = photons.n
    N_muons = muons.n
    N_all = N_prong + N_blob + N_photons + N_muons
    hist_N_mc_part = np.histogram(N_mc_part, bins=bins_N_mc_part)
    hist_N_prong = np.histogram(N_prong, bins=bins_N_prong)
    hist_N_blob = np.histogram(N_blob, bins=bins_N_blob)
    hist_N_photons = np.histogram(N_photons, bins=bins_N_photons)
    hist_N_muons = np.histogram(N_muons, bins=bins_N_muons)
    hist_N_all = np.histogram(N_all, bins=bins_N_all)
    return {
        "N_mc_part": list(hist_N_mc_part), 
        "N_prong": list(hist_N_prong), 
        "N_blob": list(hist_N_blob), 
        "N_photons": list(hist_N_photons), 
        "N_muons": list(hist_N_muons), 
        "N_all": list(hist_N_all),
        # Store raw counts for statistics
        "raw_N_mc_part": N_mc_part,
        "raw_N_prong": N_prong,
        "raw_N_blob": N_blob,
        "raw_N_photons": N_photons,
        "raw_N_muons": N_muons,
        "raw_N_all": N_all
    }

def merge_histograms_dict(dict1, dict2):
    # make sure that the bin edges are the same, then add the histograms
    for key in dict1:
        if key in dict2:
            if key.startswith("raw_"):
                # Concatenate raw arrays
                dict1[key] = np.concatenate([dict1[key], dict2[key]])
            else:
                if np.all(dict1[key][1] == dict2[key][1]):
                    dict1[key][0] += dict2[key][0]
                else:
                    raise ValueError(f"Bin edges for {key} are different")
    return dict1

def print_statistics(result):
    """Print min, max, mean, median, and percentiles for each object type per dataset."""
    print("\n" + "="*80)
    print("DATASET STATISTICS: Number of Objects per Event")
    print("="*80)
    
    for playlist in sorted(result.keys()):
        print(f"\n{'='*80}")
        print(f"Playlist: {playlist}")
        print(f"{'='*80}")
        
        # Calculate statistics for each object type
        object_types = [
            ("MC Particles", "raw_N_mc_part"),
            ("Prongs", "raw_N_prong"),
            ("Blobs", "raw_N_blob"),
            ("Photons", "raw_N_photons"),
            ("Muons", "raw_N_muons"),
            ("All Objects", "raw_N_all")
        ]
        
        for obj_name, key in object_types:
            if key in result[playlist]:
                data = result[playlist][key]
                print(f"\n  {obj_name}:")
                print(f"    Min:        {np.min(data):.0f}")
                print(f"    Max:        {np.max(data):.0f}")
                print(f"    Mean:       {np.mean(data):.2f}")
                print(f"    Median:     {np.median(data):.1f}")
                print(f"    Std Dev:    {np.std(data):.2f}")
                print(f"    Percentiles:")
                print(f"      25th:     {np.percentile(data, 25):.1f}")
                print(f"      75th:     {np.percentile(data, 75):.1f}")
                print(f"      90th:     {np.percentile(data, 90):.1f}")
                print(f"      95th:     {np.percentile(data, 95):.1f}")
                print(f"      99th:     {np.percentile(data, 99):.1f}")
    
    # Print summary comparison table
    print(f"\n\n{'='*80}")
    print("SUMMARY COMPARISON TABLE")
    print(f"{'='*80}\n")
    
    print(f"{'Playlist':<12} {'Object':<15} {'Min':<6} {'Max':<6} {'Mean':<8} {'Median':<8} {'P95':<8}")
    print("-" * 80)
    
    for playlist in sorted(result.keys()):
        for obj_name, key in object_types:
            if key in result[playlist]:
                data = result[playlist][key]
                print(f"{playlist:<12} {obj_name:<15} {np.min(data):<6.0f} {np.max(data):<6.0f} "
                      f"{np.mean(data):<8.2f} {np.median(data):<8.1f} {np.percentile(data, 95):<8.1f}")
    
    print("="*80 + "\n")

# for each dataset, get the histograms for each root file
result = {}
for playlist in DATASETS:
    print("Processing playlist: ", playlist)
    result[playlist] = {}
    root_file_max = 2
    root_file_count = 0
    for root_file in os.listdir(DATASETS[playlist]):
        if root_file.endswith(".root"):
            print("Processing root file: ", root_file)
            with uproot.open(os.path.join(DATASETS[playlist], root_file)) as f:
                master_ana_dev = f["MasterAnaDev"]
                if not len(result[playlist]):
                    result[playlist] = get_histograms(master_ana_dev)
                else:
                    result[playlist] = merge_histograms_dict(result[playlist], get_histograms(master_ana_dev))
            root_file_count += 1
            if root_file_count > root_file_max:
                break

# Print statistics for all datasets
print_statistics(result)

# Save statistics to file
if not os.path.exists("out"):
    os.makedirs("out")

with open("out/dataset_statistics.txt", "w") as f:
    f.write("="*80 + "\n")
    f.write("DATASET STATISTICS: Number of Objects per Event\n")
    f.write("="*80 + "\n")
    
    for playlist in sorted(result.keys()):
        f.write(f"\n{'='*80}\n")
        f.write(f"Playlist: {playlist}\n")
        f.write(f"{'='*80}\n")
        
        object_types = [
            ("MC Particles", "raw_N_mc_part"),
            ("Prongs", "raw_N_prong"),
            ("Blobs", "raw_N_blob"),
            ("Photons", "raw_N_photons"),
            ("Muons", "raw_N_muons"),
            ("All Objects", "raw_N_all")
        ]
        
        for obj_name, key in object_types:
            if key in result[playlist]:
                data = result[playlist][key]
                f.write(f"\n  {obj_name}:\n")
                f.write(f"    Min:        {np.min(data):.0f}\n")
                f.write(f"    Max:        {np.max(data):.0f}\n")
                f.write(f"    Mean:       {np.mean(data):.2f}\n")
                f.write(f"    Median:     {np.median(data):.1f}\n")
                f.write(f"    Std Dev:    {np.std(data):.2f}\n")
                f.write(f"    Percentiles:\n")
                f.write(f"      25th:     {np.percentile(data, 25):.1f}\n")
                f.write(f"      75th:     {np.percentile(data, 75):.1f}\n")
                f.write(f"      90th:     {np.percentile(data, 90):.1f}\n")
                f.write(f"      95th:     {np.percentile(data, 95):.1f}\n")
                f.write(f"      99th:     {np.percentile(data, 99):.1f}\n")
    
    # Summary comparison table
    f.write(f"\n\n{'='*80}\n")
    f.write("SUMMARY COMPARISON TABLE\n")
    f.write(f"{'='*80}\n\n")
    
    f.write(f"{'Playlist':<12} {'Object':<15} {'Min':<6} {'Max':<6} {'Mean':<8} {'Median':<8} {'P95':<8}\n")
    f.write("-" * 80 + "\n")
    
    for playlist in sorted(result.keys()):
        for obj_name, key in object_types:
            if key in result[playlist]:
                data = result[playlist][key]
                f.write(f"{playlist:<12} {obj_name:<15} {np.min(data):<6.0f} {np.max(data):<6.0f} "
                        f"{np.mean(data):<8.2f} {np.median(data):<8.1f} {np.percentile(data, 95):<8.1f}\n")
    
    f.write("="*80 + "\n")

print(f"\n✓ Statistics saved to out/dataset_statistics.txt\n")

fig_n_objects, ax_n_objects = plt.subplots(3, 2, figsize=(12, 15))
# Plot the histograms for each dataset
for playlist in result:
    # Normalize histograms to density (sum = 1)
    mc_part_density = result[playlist]["N_mc_part"][0] / np.sum(result[playlist]["N_mc_part"][0])
    prong_density = result[playlist]["N_prong"][0] / np.sum(result[playlist]["N_prong"][0])
    blob_density = result[playlist]["N_blob"][0] / np.sum(result[playlist]["N_blob"][0])
    photons_density = result[playlist]["N_photons"][0] / np.sum(result[playlist]["N_photons"][0])
    muons_density = result[playlist]["N_muons"][0] / np.sum(result[playlist]["N_muons"][0])
    all_density = result[playlist]["N_all"][0] / np.sum(result[playlist]["N_all"][0])
    
    ax_n_objects[0, 0].stairs(mc_part_density, result[playlist]["N_mc_part"][1], label=playlist, alpha=0.5)
    ax_n_objects[0, 1].stairs(prong_density, result[playlist]["N_prong"][1], label=playlist, alpha=0.5)
    ax_n_objects[1, 0].stairs(blob_density, result[playlist]["N_blob"][1], label=playlist, alpha=0.5)
    ax_n_objects[1, 1].stairs(photons_density, result[playlist]["N_photons"][1], label=playlist, alpha=0.5)
    ax_n_objects[2, 0].stairs(muons_density, result[playlist]["N_muons"][1], label=playlist, alpha=0.5)
    ax_n_objects[2, 1].stairs(all_density, result[playlist]["N_all"][1], label=playlist, alpha=0.5)

ax_n_objects[0, 0].legend()
ax_n_objects[0, 1].legend()
ax_n_objects[1, 0].legend()
ax_n_objects[1, 1].legend()
ax_n_objects[2, 0].legend()
ax_n_objects[2, 1].legend()
ax_n_objects[0, 0].set_xlabel("Number of MC Particles per Event")
ax_n_objects[0, 0].set_ylabel("Density")
ax_n_objects[0, 0].set_yscale("log")
ax_n_objects[0, 0].set_title("MC Particle Multiplicity Distribution")
ax_n_objects[0, 1].set_xlabel("Number of Prongs per Event")
ax_n_objects[0, 1].set_ylabel("Density")
ax_n_objects[0, 1].set_title("Prong Multiplicity Distribution")
ax_n_objects[1, 0].set_xlabel("Number of Blobs per Event")
ax_n_objects[1, 0].set_yscale("log")
ax_n_objects[1, 0].set_ylabel("Density")
ax_n_objects[1, 0].set_title("Blob Multiplicity Distribution")
ax_n_objects[1, 1].set_xlabel("Number of Photons per Event")
ax_n_objects[1, 1].set_ylabel("Density")
ax_n_objects[1, 1].set_title("Photon Multiplicity Distribution")
ax_n_objects[2, 0].set_xlabel("Number of Muons per Event")
ax_n_objects[2, 0].set_ylabel("Density")
ax_n_objects[2, 0].set_title("Muon Multiplicity Distribution")
ax_n_objects[2, 1].set_xlabel("Number of Objects per Event")
ax_n_objects[2, 1].set_ylabel("Density")
ax_n_objects[2, 1].set_title("Object Multiplicity Distribution")
ax_n_objects[2, 1].set_yscale("log")
fig_n_objects.tight_layout()
fig_n_objects.savefig("n_objects.pdf")
#fig_n_objects.show()

if not os.path.exists("out"):
    os.makedirs("out")
fig_n_objects.savefig("out/n_objects.pdf")

