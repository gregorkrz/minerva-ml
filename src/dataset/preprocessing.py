from dataclasses import dataclass
import numpy as np
import awkward as ak
import os
import h5py
import torch


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
    matrix[np.isnan(matrix)] = 0
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
    photon_four_vectors[np.isnan(photon_four_vectors)] = 0
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
    muon_matrix[np.isnan(muon_matrix)] = 0
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

def convert_four_momentum(four_momentum): # convert [px, py, pz, E] to eta, phi, log(pT), log(E)
    """
    Convert 4-momentum [px, py, pz, E] to OmniLearned features [Δη, Δφ, log(pT), log(E)].
    
    Args:
        four_momentum: array of shape (n_particles, 4) with [px, py, pz, E]
    
    Returns:
        features: array of shape (n_particles, 4) with [Δη, Δφ, log(pT), log(E)]
    """
    # Extract momentum components
    px = four_momentum[:, 0]
    py = four_momentum[:, 1]
    pz = four_momentum[:, 2]
    E = four_momentum[:, 3]
    
    # Compute kinematic variables
    pt = np.sqrt(px**2 + py**2)
    p = np.sqrt(px**2 + py**2 + pz**2)
    
    # Pseudorapidity: η = -ln(tan(θ/2)) = 0.5 * ln((|p| + pz) / (|p| - pz))
    eta = np.zeros_like(pz)
    # Need to check both numerator and denominator are not too small
    valid = (p > 1e-6) & (np.abs(p + pz) > 1e-6) & (np.abs(p - pz) > 1e-6)
    # Clip eta to reasonable range to avoid infinities
    eta[valid] = np.clip(0.5 * np.log((p[valid] + pz[valid]) / (p[valid] - pz[valid])), -10, 10)
    
    # Azimuthal angle
    phi = np.arctan2(py, px)
    
    # For neutrino interactions, use beam axis (z-direction) as reference
    delta_eta = eta
    delta_phi = phi
    
    # Log transforms (add small epsilon to avoid log(0))
    if not np.all(pt >= 0):
        print("pt issues", pt[~ (pt>=0)])
    assert np.all(pt >= 0)
    assert np.all(E >= 0)

    log_pt = np.log(pt + 1e-6)
    log_E = np.log(E + 1e-6)
    
    # Stack features: [Δη, Δφ, log(pT), log(E)]
    features = np.stack([delta_eta, delta_phi, log_pt, log_E], axis=1)
    return features


def get_event_repr_nested_tensor(muons, photons, blobs, prongs, global_features, truth_labels, output_file, max_objects=150):
    """
    Convert event data to a nested tensor format.
    
    Args:
        muons: DenseCollection of muon data
        photons: DenseCollection of photon data
        blobs: DenseCollection of blob data
        prongs: DenseCollection of prong data
        global_features: Global, event-level features
        incoming_E: Incoming neutrino energy (Ground Truth)
        event_type: Event type (Ground Truth)
        output_file: Path to output HDF5 file
        max_objects: Maximum number of objects per event
        chunk_size: Number of events to process at once before writing to disk
    Returns: Number of events written to file

    """
    muon_four_momentum = muons.data[:, :4]
    photon_four_momentum = photons.data[:, :4]
    blob_four_momentum = blobs.data[:, [0, 1, 2, 5]].copy() # really this is [x, y, z, E]
    # Adjust blob x, y, z such that we assume a massless particle originating from the origin
    # Convert position (x,y,z) to momentum-like (px,py,pz) by normalizing direction and scaling by energy
    blob_norm = np.linalg.norm(blob_four_momentum[:, 0:3], axis=1, keepdims=True)
    blob_norm = np.where(blob_norm > 1e-6, blob_norm, 1.0)  # Avoid division by zero
    blob_four_momentum[:, 0:3] = (blob_four_momentum[:, 0:3] / blob_norm) * blob_four_momentum[:, 3:4]
    prong_four_momentum = prongs.data[:, 4:8]
    prong_PID = prongs.data[:, -1]
    # Our PID hardcoded:
    # muon = 0, photon = 1, blob = 2 (no PID), prong PIDs: -999=3, 0=4, 3=5, 4=6, 5=7
    # Convert prong_pid according to the above mapping - for now, hardcoded
    prong_pid = np.zeros(len(prong_PID), dtype=np.int32)
    for i in range(len(prong_PID)):
        if prong_PID[i] == -999:
            prong_pid[i] = 3
        elif prong_PID[i] == 0:
            prong_pid[i] = 4
        elif prong_PID[i] == 3:
            prong_pid[i] = 5
        elif prong_PID[i] == 4:
            prong_pid[i] = 6
    muon_pid = np.zeros(len(muon_four_momentum), dtype=np.int32)
    photon_pid = np.ones(len(photon_four_momentum), dtype=np.int32)
    blob_pid = np.ones(len(blob_four_momentum), dtype=np.int32) * 2 # Not real PID!
    
    # Convert all features at once (more efficient)
    muon_features = np.concatenate([convert_four_momentum(muon_four_momentum), muon_pid[:, np.newaxis]], axis=1)
    photon_features = np.concatenate([convert_four_momentum(photon_four_momentum), photon_pid[:, np.newaxis]], axis=1)
    blob_features = np.concatenate([convert_four_momentum(blob_four_momentum), blob_pid[:, np.newaxis]], axis=1)
    prong_features = np.concatenate([convert_four_momentum(prong_four_momentum), prong_pid[:, np.newaxis]], axis=1)

    # Get number of events
    N_events = len(muons.n)
    assert N_events == len(photons.n) == len(blobs.n) == len(prongs.n)
    assert len(global_features) == N_events, f"Global features length {len(global_features)} doesn't match N_events {N_events}"
    
    file_exists = os.path.exists(output_file)
    data_nested = []

    if file_exists:
        raise ValueError(f"File {output_file} already exists")
    for event_idx in range(N_events):
        if event_idx % 1000 == 0:
            print(f"Processed events {event_idx} to {event_idx+1000} / {N_events}")
        muon_features_event = muon_features[muons.bounds[event_idx]:muons.bounds[event_idx+1]]
        photon_features_event = photon_features[photons.bounds[event_idx]:photons.bounds[event_idx+1]]
        blob_features_event = blob_features[blobs.bounds[event_idx]:blobs.bounds[event_idx+1]]
        prong_features_event = prong_features[prongs.bounds[event_idx]:prongs.bounds[event_idx+1]]
        event_features = np.concatenate([muon_features_event, photon_features_event, blob_features_event, prong_features_event], axis=0)
        if len(event_features) > max_objects:
            # Sort by energy (descending)
            event_features_idx_energy = event_features[:, 3].argsort()[::-1]
            # Keep only the max_objects per event
            n_to_keep = min(len(event_features), max_objects)
            event_features = event_features[event_features_idx_energy[:n_to_keep]]
        data_nested.append(event_features)
    data_nested = torch.nested.nested_tensor(data_nested, layout=torch.jagged)
    torch.save({"data": data_nested, "truth_labels": truth_labels, "global_features": global_features}, output_file)
    print(f"✓ Nested tensor data written to {output_file} (total events: {N_events})")
    return N_events


def get_event_repr(muons, photons, blobs, prongs, global_features, truth_labels, output_file, max_objects=100, chunk_size=1000):
    """
    Convert event data to OmniLearned format and write directly to HDF5 file.
    
    Args:
        muons: DenseCollection of muon data
        photons: DenseCollection of photon data
        blobs: DenseCollection of blob data
        prongs: DenseCollection of prong data
        global_features: Global, event-level features
        incoming_E: Incoming neutrino energy (Ground Truth)
        event_type: Event type (Ground Truth)
        output_file: Path to output HDF5 file
        max_objects: Maximum number of objects per event
        chunk_size: Number of events to process at once before writing to disk
    Returns: Number of events written to file

    """
    muon_four_momentum = muons.data[:, :4]
    photon_four_momentum = photons.data[:, :4]
    blob_four_momentum = blobs.data[:, [0, 1, 2, 5]].copy() # really this is [x, y, z, E]
    # Adjust blob x, y, z such that we assume a massless particle originating from the origin
    # Convert position (x,y,z) to momentum-like (px,py,pz) by normalizing direction and scaling by energy
    blob_norm = np.linalg.norm(blob_four_momentum[:, 0:3], axis=1, keepdims=True)
    blob_norm = np.where(blob_norm > 1e-6, blob_norm, 1.0)  # Avoid division by zero
    blob_four_momentum[:, 0:3] = (blob_four_momentum[:, 0:3] / blob_norm) * blob_four_momentum[:, 3:4]
    prong_four_momentum = prongs.data[:, 4:8]
    prong_PID = prongs.data[:, -1]
    # Our PID hardcoded:
    # muon = 0, photon = 1, blob = 2 (no PID), prong PIDs: -999=3, 0=4, 3=5, 4=6, 5=7
    # Convert prong_pid according to the above mapping - for now, hardcoded
    prong_pid = np.zeros(len(prong_PID), dtype=np.int32)
    for i in range(len(prong_PID)):
        if prong_PID[i] == -999:
            prong_pid[i] = 3
        elif prong_PID[i] == 0:
            prong_pid[i] = 4
        elif prong_PID[i] == 3:
            prong_pid[i] = 5
        elif prong_PID[i] == 4:
            prong_pid[i] = 6
    muon_pid = np.zeros(len(muon_four_momentum), dtype=np.int32)
    photon_pid = np.ones(len(photon_four_momentum), dtype=np.int32)
    blob_pid = np.ones(len(blob_four_momentum), dtype=np.int32) * 2 # Not real PID!
    # Convert all features at once (more efficient)
    muon_features = np.concatenate([convert_four_momentum(muon_four_momentum), muon_pid[:, np.newaxis]], axis=1)
    photon_features = np.concatenate([convert_four_momentum(photon_four_momentum), photon_pid[:, np.newaxis]], axis=1)
    blob_features = np.concatenate([convert_four_momentum(blob_four_momentum), blob_pid[:, np.newaxis]], axis=1)
    prong_features = np.concatenate([convert_four_momentum(prong_four_momentum), prong_pid[:, np.newaxis]], axis=1)

    # Get number of events
    N_events = len(muons.n)
    assert N_events == len(photons.n) == len(blobs.n) == len(prongs.n)
    assert len(global_features) == N_events, f"Global features length {len(global_features)} doesn't match N_events {N_events}"
    
    # Get number of global features
    n_global_features = global_features.shape[1] if len(global_features.shape) > 1 else 1
    if len(global_features.shape) == 1:
        global_features = global_features[:, np.newaxis]
    

    file_exists = os.path.exists(output_file)
    existing_events = 0
    
    if file_exists:
        with h5py.File(output_file, 'r') as hf:
            existing_events = hf['data'].shape[0]
            print(f"File exists with {existing_events} events, appending {N_events} new events...")
    else:
        print(f"Creating new file with {N_events} events...")
    
    # Open file in appropriate mode
    mode = 'a' if file_exists else 'w'
    
    # Create or append to HDF5 file
    with h5py.File(output_file, mode) as hf:
        if file_exists:
            # Resize existing datasets to accommodate new events
            dset = hf['data']
            dset_global = hf['global']
            dset_truth_labels = hf['truth_labels']
            dset_number_of_particles = hf['number_of_particles']
            old_size = dset.shape[0]
            new_size = old_size + N_events
            dset.resize(new_size, axis=0)
            dset_global.resize(new_size, axis=0)
            dset_truth_labels.resize(new_size, axis=0)
            dset_number_of_particles.resize(new_size, axis=0)
            write_offset = old_size
        else:
            # Create datasets with chunking for efficient writing
            dset = hf.create_dataset(
                'data', 
                shape=(N_events, max_objects, 5),
                maxshape=(None, max_objects, 5),  # Allow unlimited growth in first dimension
                dtype=np.float32,
                chunks=(min(chunk_size, N_events), max_objects, 5),
                compression='gzip',
                compression_opts=4
            )
            dset_global = hf.create_dataset(
                'global',
                shape=(N_events, n_global_features),
                maxshape=(None, n_global_features),
                dtype=np.float32,
                chunks=(min(chunk_size, N_events), n_global_features),
                compression='gzip',
                compression_opts=4
            )
            dset_truth_labels = hf.create_dataset(
                'truth_labels',
                shape=(N_events, 3),
                maxshape=(None, 3),
                dtype=np.float32,
                chunks=(min(chunk_size, N_events), 3),
                compression='gzip',
                compression_opts=4
            )
            dset_number_of_particles = hf.create_dataset(
                'number_of_particles',
                shape=(N_events,),
                maxshape=(None,),
                dtype=np.int32,
                chunks=(min(chunk_size, N_events),),
                compression='gzip',
                compression_opts=4
            ) # Number of tokens per event - to make it easier to plot histograms of quantities
            write_offset = 0
        # Process events in chunks to save memory
        for chunk_start in range(0, N_events, chunk_size):
            chunk_end = min(chunk_start + chunk_size, N_events)
            chunk_data = np.zeros((chunk_end - chunk_start, max_objects, 5), dtype=np.float32)
            n_particles_in_chunk = np.zeros(chunk_end - chunk_start, dtype=np.int32)
            for i, event_idx in enumerate(range(chunk_start, chunk_end)):
                muon_features_event = muon_features[muons.bounds[event_idx]:muons.bounds[event_idx+1]]
                photon_features_event = photon_features[photons.bounds[event_idx]:photons.bounds[event_idx+1]]
                blob_features_event = blob_features[blobs.bounds[event_idx]:blobs.bounds[event_idx+1]]
                prong_features_event = prong_features[prongs.bounds[event_idx]:prongs.bounds[event_idx+1]]
                # Stack them all into a single array
                event_features = np.concatenate([muon_features_event, photon_features_event, blob_features_event, prong_features_event], axis=0)
                if len(event_features) > 0:
                    # Sort by energy (descending)
                    event_features_idx_energy = event_features[:, 3].argsort()[::-1]
                    # Keep only the max_objects per event
                    n_to_keep = min(len(event_features), max_objects)
                    event_features_sorted = event_features[event_features_idx_energy[:n_to_keep]]
                    chunk_data[i, :n_to_keep] = event_features_sorted
                    n_particles_in_chunk[i] = n_to_keep  # Store number actually kept
                else:
                    n_particles_in_chunk[i] = 0
            # Write chunk to disk at the appropriate offset
            dset[write_offset + chunk_start:write_offset + chunk_end] = chunk_data
            # Write truth labels
            dset_truth_labels[write_offset + chunk_start:write_offset + chunk_end] = truth_labels[chunk_start:chunk_end]
            dset_number_of_particles[write_offset + chunk_start:write_offset + chunk_end] = n_particles_in_chunk
            print(f"Processed events {chunk_start} to {chunk_end} / {N_events} (total in file: {write_offset + chunk_end})")
        # Write global features
        dset_global[write_offset:write_offset + N_events] = global_features
        # Update metadata
        total_events = write_offset + N_events
        hf.attrs['n_events'] = total_events
        hf.attrs['max_objects'] = max_objects
        hf.attrs['n_features'] = 5
        hf.attrs['n_global_features'] = n_global_features
        hf.attrs['feature_names'] = ['delta_eta', 'delta_phi', 'log_pt', 'log_E', 'pid']
        hf.attrs['global_feature_names'] = ['muon_fuzz_energy', 'muon_iso_blobs_energy', 'E_recoil', 'E_recoil_CCinc']
        hf.attrs['pid_mapping'] = 'muon=0, photon=1, blob=2, prong_-999=3, prong_0=4, prong_3=5, prong_4=6'
    print(f"✓ Data written to {output_file} (total events: {total_events})")
    return N_events


def get_global_features(master_ana_dev):
    # Get the global features: muon fuzz energy, muon iso blobs energy, E_recoil, E_recoil_CCinc
    muon_fuzz_energy = master_ana_dev["muon_fuzz_energy"].array().to_numpy()
    muon_iso_blobs_energy = master_ana_dev["muon_iso_blobs_energy"].array().to_numpy()
    muon_fuzz_energy[muon_fuzz_energy < 0] = 0
    muon_iso_blobs_energy[muon_iso_blobs_energy < 0] = 0
    E_recoil = master_ana_dev["MasterAnaDev_hadron_recoil"].array().to_numpy()
    E_recoil_CCinc = master_ana_dev["MasterAnaDev_hadron_recoil_CCInc"].array().to_numpy()
    E_recoil[E_recoil < 0] = 0
    E_recoil_CCinc[E_recoil_CCinc < 0] = 0
    # stack them together, so that the shape is (n_events, 4)
    global_features = np.stack([muon_fuzz_energy, muon_iso_blobs_energy, E_recoil, E_recoil_CCinc], axis=1)
    # compute log10 of the global features + 1e-5 and store that
    global_features = np.log(global_features + 1e-5)
    return global_features

def get_event_labels(master_ana_dev):
    incoming_E = master_ana_dev["mc_incomingE"].array().to_numpy()
    event_type = master_ana_dev["mc_intType"].array().to_numpy()
    muon_reco_energy = master_ana_dev["muon_corrected_p"].array().to_numpy()[:, 3]
    bad_muons = muon_reco_energy < 0
    E_nu_minus_muon_reco_energy = incoming_E - muon_reco_energy
    E_nu_minus_muon_reco_energy[bad_muons] = -1
    return np.stack([incoming_E, event_type, E_nu_minus_muon_reco_energy], axis=1)

def get_event_collections(master_ana_dev):
    mc_part_keys = ["mc_FSPartPx", "mc_FSPartPy", "mc_FSPartPz", "mc_FSPartE", "mc_FSPartPDG"]
    prong_keys = ["prong_part_pos", "prong_part_E", "prong_part_score", "prong_part_mass", "prong_part_charge", "prong_part_pid"]
    blob_keys = ["MasterAnaDev_BlobX", "MasterAnaDev_BlobY", "MasterAnaDev_BlobZ", "MasterAnaDev_BlobT", "MasterAnaDev_BlobTPos", "MasterAnaDev_BlobTotalE"]
    mc_part = get_dense(mc_part_keys, master_ana_dev)
    prong = get_dense(prong_keys, master_ana_dev)
    blob = get_dense(blob_keys, master_ana_dev)
    muons = get_muons(master_ana_dev)
    photons = get_photons(master_ana_dev)
    muons = remove_overflows(muons)
    photons = remove_overflows(photons)
    # remove overflows
    mc_part = remove_overflows(mc_part)
    prong = remove_overflows(prong)
    blob = remove_overflows(blob)
    return mc_part, prong, blob, muons, photons
