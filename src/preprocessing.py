from dataclasses import dataclass
import numpy as np
import awkward as ak

@dataclass
class DenseCollection:
    bounds: np.ndarray  # shape (n_events + 1,)
    data: np.ndarray    # shape (n_total_particles, n_features)
    n: np.ndarray       # shape (n_events,) - simply the one-difference of bounds
    keys: list = None
    column_widths: list = None

def get_dense(keys, master_ana_dev_frame):
    column_arrays = [master_ana_dev_frame[key].array() for key in keys]
    # Get number of particles per event from mc_part_arrays[0]
    n_per_event = ak.num(column_arrays[0])
    column_arrays = [ak.flatten(arr).to_numpy() for arr in column_arrays]
    column_widths = []
    for i in column_arrays:
        if len(i.shape) == 1:
            column_widths.append(1)
        else:
            column_widths.append(i.shape[1])
    # Now, put the mc_part_arrays in a dense matrix + make an index tensor where i and i+1 elements point to event boundaries of ith event
    matrix = np.zeros((len(column_arrays[0]), sum(column_widths)), dtype = np.float32)
    # Replace the nans with zeros
    start_idx = 0
    for i, arr in enumerate(column_arrays):
        if column_widths[i] == 1:
            matrix[:, start_idx] = arr
        else:
            matrix[:, start_idx:start_idx+column_widths[i]] = arr
        start_idx += column_widths[i]
    event_boundaries = np.zeros(len(n_per_event) + 1, dtype=np.int32)
    event_boundaries[1:] = np.cumsum(n_per_event)
    return DenseCollection(bounds=event_boundaries, data=matrix, n=n_per_event.to_numpy(), keys=keys, column_widths=column_widths)

def get_photons(master_ana_dev_frame):
    # slightly special treatment for gamma1 and gamma2
    gamma1_keys = ["gamma1_px", "gamma1_py", "gamma1_pz", "gamma1_E"]
    gamma2_keys = ["gamma2_px", "gamma2_py", "gamma2_pz", "gamma2_E"]
    
    # Get gamma1 and gamma2 arrays
    gamma_arrays_1 = [master_ana_dev_frame[key].array() for key in gamma1_keys]
    gamma_arrays_2 = [master_ana_dev_frame[key].array() for key in gamma2_keys]
    
    # Filter for valid photons (E > 1e-5)
    filter_1 = gamma_arrays_1[3] > 1e-5
    filter_2 = gamma_arrays_2[3] > 1e-5

    gamma_arrays_1 = [arr[filter_1].to_numpy() for arr in gamma_arrays_1]
    gamma_arrays_2 = [arr[filter_2].to_numpy() for arr in gamma_arrays_2]
    gamma_arrays_1 = np.stack(gamma_arrays_1).T
    gamma_arrays_2 = np.stack(gamma_arrays_2).T
    
    # Count photons per event
    n_gamma1_per_event = filter_1.to_numpy().astype(np.int32)
    n_gamma2_per_event = filter_2.to_numpy().astype(np.int32)
    n_per_event = n_gamma1_per_event + n_gamma2_per_event
    
    # Build photon data in event order
    photon_four_vectors = np.zeros((sum(n_per_event), 4), dtype=np.float32)
    current_idx = 0
    current_idx_gamma1 = 0
    current_idx_gamma2 = 0
    for i in range(len(n_per_event)):
        photon_four_vectors[current_idx:current_idx+n_gamma1_per_event[i], :] = gamma_arrays_1[current_idx_gamma1:current_idx_gamma1+n_gamma1_per_event[i]]
        current_idx += n_gamma1_per_event[i]
        # the line below fails, print debugging info - indices, ...
        photon_four_vectors[current_idx:current_idx+n_gamma2_per_event[i], :] = gamma_arrays_2[current_idx_gamma2:current_idx_gamma2+n_gamma2_per_event[i]]
        current_idx_gamma1 += n_gamma1_per_event[i]
        current_idx_gamma2 += n_gamma2_per_event[i]
        current_idx += n_gamma2_per_event[i]
    # Create event boundaries
    event_boundaries = np.zeros(len(n_per_event) + 1, dtype=np.int32)
    event_boundaries[1:] = np.cumsum(n_per_event)
    return DenseCollection(bounds=event_boundaries, data=photon_four_vectors, n=n_per_event, keys=gamma1_keys, column_widths=[1, 1, 1, 1])

def get_muons(master_ana_dev_frame, only_keep_minos_matched=True):
    # Slightly special treatment for muons
    muon_keys = ["muon_corrected_p", "muon_theta", "muon_phi", "muon_thetaX", "muon_thetaY", "muon_fuzz_energy", "muon_iso_blobs_energy", "MasterAnaDev_muon_E", "MasterAnaDev_muon_Px", "MasterAnaDev_muon_Py", "MasterAnaDev_muon_Pz"]
    # filter out cases where muon_corrected_p[:, 0] == -999 and muon_corrected_p[:, 1]] == -999
    muon_arrays = [master_ana_dev_frame[key].array() for key in muon_keys]
    if only_keep_minos_matched:
        mask = (muon_arrays[0][:, 0] != -999) & (muon_arrays[0][:, 1] != -999)
    else:
        #mask = master_ana_dev_frame["truth_reco_has_muon"].array()
        mask = np.ones(len(muon_arrays[0]), dtype=bool)
        mask = ak.from_numpy(mask)
    muon_arrays = [arr[mask].to_numpy() for arr in muon_arrays]
    # Exclude muons with abs of any component of any key > 1e6 - some weird anomalies
    n_muon_per_event = mask.to_numpy().astype(np.int32)
    # now put the muon_arrays in a dense matrix + make an index tensor where i and i+1 elements point to event boundaries of ith event
    muon_matrix = np.zeros((len(muon_arrays[0]), len(muon_keys)+3), dtype=np.float32)
    column_widths = [4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    start_idx = 0
    for i, arr in enumerate(muon_arrays):
        if column_widths[i] == 1:
            muon_matrix[:, start_idx] = arr
        else:
            muon_matrix[:, start_idx:start_idx+column_widths[i]] = arr
        start_idx += column_widths[i]
    muon_event_boundaries = np.zeros(len(n_muon_per_event) + 1, dtype=np.int32)
    muon_event_boundaries[1:] = np.cumsum(n_muon_per_event)
    # Use: mc_part_event_boundaries and mc_part_matrix
    return DenseCollection(bounds=muon_event_boundaries, data=muon_matrix, n=n_muon_per_event, keys=muon_keys, column_widths=column_widths)

def remove_overflows(coll: DenseCollection):
    # Some entries in the data matrix are weird. Remove large numbers
    data_anomalies = np.where(np.abs(coll.data).max(axis=1) > 1e6)[0]
    data_anomalies_set = set(list(data_anomalies))
    n_per_event = coll.n.copy()
    data_filtered = np.delete(coll.data, data_anomalies, axis=0)
    event_bounds_old = coll.bounds
    # go through event bounds and create a new event bounds without the removed particles.
    #  shape of n_per_event should stay the same, just some events will have less (or 0) particles after removing.
    event_bounds_new = np.zeros(len(n_per_event) + 1, dtype=np.int32)
    n_per_event_new = np.zeros(len(n_per_event), dtype=np.int32)
    
    for event_idx in range(len(n_per_event)):
        # Get the particle indices for this event
        start_particle_idx = event_bounds_old[event_idx]
        end_particle_idx = event_bounds_old[event_idx + 1]
        
        # Count how many particles in this event are NOT anomalies
        n_removed = 0
        for particle_idx in range(start_particle_idx, end_particle_idx):
            if particle_idx in data_anomalies_set:
                n_removed += 1
        
        # Update the new particle count for this event
        n_per_event_new[event_idx] = n_per_event[event_idx] - n_removed
        # Update the event bounds
        event_bounds_new[event_idx + 1] = event_bounds_new[event_idx] + n_per_event_new[event_idx]
    
    return DenseCollection(bounds=event_bounds_new, data=data_filtered, n=n_per_event_new, keys=coll.keys, column_widths=coll.column_widths)


def get_event_repr(muons, photons, blobs, prongs):
    muon_four_momentum = muons.data[:, :4]
    photon_four_momentum = photons.data[:, :4]
    blob_four_momentum = blobs.data[:, :4]
    prong_four_momentum = prongs.data[:, 4:8]
    prong_PID = prongs.data[:, -1]
    # Our PID hardcoded:
    # muon = 0, photon = 1, blob = 2 (no PID), prong PIDs: -999=3, 0=4, 3=5, 4=6, 5=7
