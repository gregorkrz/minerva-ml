from pathlib import Path
import ROOT
import matplotlib.pyplot as plt
import numpy as np
import os
import uproot
import awkward as ak
from src.resolution_tools import find_narrowest_interval
from src.preprocessing import get_event_repr, get_muons, get_photons, get_dense, remove_overflows, get_global_features, get_event_labels

DATASETS = {}
#for playlist in ["1A", "1B", "1C", "1D", "1E", "1F", "1G", "1L", "1M", "1N", "1O", "1P"]:
for playlist in ["1A", "1B"]:
    DATASETS[playlist] = f"/scratch/MINERvA/raw_data/MediumEnergy_FHC_StandardMC_Playlist/{playlist}"
MAX_OBJECTS = 100

mc_part_keys = ["mc_FSPartPx", "mc_FSPartPy", "mc_FSPartPz", "mc_FSPartE", "mc_FSPartPDG"]
prong_keys = ["prong_part_pos", "prong_part_E", "prong_part_score", "prong_part_mass", "prong_part_charge", "prong_part_pid"]
blob_keys = ["MasterAnaDev_BlobX", "MasterAnaDev_BlobY", "MasterAnaDev_BlobZ", "MasterAnaDev_BlobT", "MasterAnaDev_BlobTPos", "MasterAnaDev_BlobTotalE"]

def merge_histograms_dict(dict1, dict2):
    # make sure that the bin edges are the same, then add the histograms
    for key in dict1:
        if key in dict2:
            if np.all(dict1[key][1] == dict2[key][1]):
                dict1[key][0] += dict2[key][0]
            else:
                raise ValueError(f"Bin edges for {key} are different")
    return dict1

# For each dataset, get the histograms for each root file
result = {}
for playlist in DATASETS:
    print("Processing playlist: ", playlist)
    result[playlist] = {}
    for root_file in os.listdir(DATASETS[playlist]):
        if root_file.endswith(".root"):
            print("Processing root file: ", root_file)
            with uproot.open(os.path.join(DATASETS[playlist], root_file)) as f:
                master_ana_dev = f["MasterAnaDev"]
                muons = get_muons(master_ana_dev)
                photons = get_photons(master_ana_dev)
                blobs = get_dense(blob_keys, master_ana_dev)
                prongs = get_dense(prong_keys, master_ana_dev)
                muons = remove_overflows(muons)
                global_features = get_global_features(master_ana_dev)
                truth_labels = get_event_labels(master_ana_dev)
                get_event_repr(muons, photons, blobs, prongs, global_features, truth_labels, max_objects=MAX_OBJECTS, output_file=f"/data/events_{playlist}.h5")
