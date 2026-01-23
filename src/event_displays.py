def event_display(event_idx, muons=None, photons=None, mc_part=None, blobs=None, prongs=None, 
                  figsize=(7, 7), title=None):
    """
    Display an event in the theta-phi plane with particle energies shown as marker sizes.
    
    Parameters:
    -----------
    event_idx : int
        Event index to display
    muons, photons, mc_part, blobs, prongs : DenseCollection, optional
        Collections of particles to display
    figsize : tuple
        Figure size
    title : str, optional
        Custom title for the plot
    """
    
    def compute_theta_phi_energy(px, py, pz, E):
        """Compute theta, phi, and energy from momentum components"""
        p = np.sqrt(px**2 + py**2 + pz**2)
        theta = np.arctan2(np.sqrt(px**2 + py**2), pz)  # angle from z-axis
        phi = np.arctan2(py, px)  # angle in x-y plane
        return theta, phi, E
    
    def get_column_slice(collection, key):
        """Get the column slice for a given key in the collection"""
        start_idx = 0
        for i, k in enumerate(collection.keys):
            if k == key:
                width = collection.column_widths[i]
                if width == 1:
                    return collection.data[:, start_idx]
                else:
                    return collection.data[:, start_idx:start_idx+width]
            start_idx += collection.column_widths[i]
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # PDG code to name mapping for common particles
    pdg_names = {
        22: 'γ', 211: 'π+', -211: 'π-', 111: 'π0',
        2212: 'p', -2212: 'p̄', 2112: 'n', -2112: 'n̄',
        13: 'μ-', -13: 'μ+', 11: 'e-', -11: 'e+',
        321: 'K+', -321: 'K-', 130: 'K0L', 310: 'K0S'
    }

    pdg_names_for_prong = {
        3: "prong_μ",
        8: "prong_p"
    }
    
    collections_data = []
    
    # Process muons
    if muons is not None and muons.n[event_idx] > 0:
        start, end = muons.bounds[event_idx], muons.bounds[event_idx + 1]
        muon_data = muons.data[start:end]
        
        # Get momentum components - check if using MasterAnaDev_muon or corrected_p
        if 'MasterAnaDev_muon_Px' in muons.keys:
            px = get_column_slice(muons, 'MasterAnaDev_muon_Px')[start:end]
            py = get_column_slice(muons, 'MasterAnaDev_muon_Py')[start:end]
            pz = get_column_slice(muons, 'MasterAnaDev_muon_Pz')[start:end]
            E = get_column_slice(muons, 'MasterAnaDev_muon_E')[start:end]
        else:
            # Compute from angles if available
            p = get_column_slice(muons, 'muon_corrected_p')[start:end, 0]
            theta_x = get_column_slice(muons, 'muon_thetaX')[start:end]
            theta_y = get_column_slice(muons, 'muon_thetaY')[start:end]
            px = p * np.sin(theta_x)
            py = p * np.sin(theta_y)
            pz = p * np.sqrt(1 - np.sin(theta_x)**2 - np.sin(theta_y)**2)
            E = get_column_slice(muons, 'MasterAnaDev_muon_E')[start:end]
        
        theta, phi, energy = compute_theta_phi_energy(px, py, pz, E)
        collections_data.append(('Muons', theta, phi, energy, 'blue', 'o', None))
    
    # Process photons
    if photons is not None and photons.n[event_idx] > 0:
        start, end = photons.bounds[event_idx], photons.bounds[event_idx + 1]
        px = get_column_slice(photons, 'gamma1_px')[start:end]
        py = get_column_slice(photons, 'gamma1_py')[start:end]
        pz = get_column_slice(photons, 'gamma1_pz')[start:end]
        E = get_column_slice(photons, 'gamma1_E')[start:end]
        
        theta, phi, energy = compute_theta_phi_energy(px, py, pz, E)
        collections_data.append(('Photons', theta, phi, energy, 'red', '*', None))
    
    # Process MC particles
    if mc_part is not None and mc_part.n[event_idx] > 0:
        start, end = mc_part.bounds[event_idx], mc_part.bounds[event_idx + 1]
        px = get_column_slice(mc_part, 'mc_FSPartPx')[start:end]
        py = get_column_slice(mc_part, 'mc_FSPartPy')[start:end]
        pz = get_column_slice(mc_part, 'mc_FSPartPz')[start:end]
        E = get_column_slice(mc_part, 'mc_FSPartE')[start:end]
        pdg = get_column_slice(mc_part, 'mc_FSPartPDG')[start:end].astype(int)
        
        theta, phi, energy = compute_theta_phi_energy(px, py, pz, E)
        collections_data.append(('MC Particles', theta, phi, energy, 'green', 's', pdg))
    
    # Process blobs
    if blobs is not None and blobs.n[event_idx] > 0:
        start, end = blobs.bounds[event_idx], blobs.bounds[event_idx + 1]
        x = get_column_slice(blobs, 'MasterAnaDev_BlobX')[start:end]
        y = get_column_slice(blobs, 'MasterAnaDev_BlobY')[start:end]
        z = get_column_slice(blobs, 'MasterAnaDev_BlobZ')[start:end]
        E = get_column_slice(blobs, 'MasterAnaDev_BlobTotalE')[start:end]
        
        # Treat position as direction
        theta, phi, energy = compute_theta_phi_energy(x, y, z, E)
        collections_data.append(('Blobs', theta, phi, energy, 'orange', 'D', None))
    
    # Process prongs
    if prongs is not None and prongs.n[event_idx] > 0:
        start, end = prongs.bounds[event_idx], prongs.bounds[event_idx + 1]
        #pos = get_column_slice(prongs, 'prong_part_pos')[start:end]
        pos = get_column_slice(prongs, 'prong_part_E')[start:end][:, :3]
        E = get_column_slice(prongs, 'prong_part_E')[start:end][:, -1]
        pid = get_column_slice(prongs, 'prong_part_pid')[start:end].astype(int)
        
        # Extract x, y, z from position (assuming it's 3D)
        if len(pos.shape) == 2 and pos.shape[1] >= 3:
            x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
            theta, phi, energy = compute_theta_phi_energy(x, y, z, E)
            collections_data.append(('Prongs', theta, phi, energy, 'purple', '^', pid))
    
    # Plot all collections
    for name, theta, phi, energy, color, marker, labels in collections_data:
        # Size proportional to energy (scale for visibility)
        sizes = (energy / np.max(energy) * 500) if len(energy) > 0 and np.max(energy) > 0 else 100
        scatter = ax.scatter(phi, theta, s=sizes, c=color, marker=marker, 
                           alpha=0.6, edgecolors='black', linewidths=0.5, label=name)
        
        # Add labels for particles with PID information
        if labels is not None:
            for i, (p, t, lbl) in enumerate(zip(phi, theta, labels)):
                if name == "Prongs":
                    label_text = pdg_names_for_prong.get(lbl, str(lbl))
                else:
                    label_text = pdg_names.get(lbl, str(lbl))
                ax.annotate(label_text, (p, t), fontsize=8, ha='center', va='center',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
    
    ax.set_xlabel('φ (radians)', fontsize=12)
    ax.set_ylabel('θ (radians)', fontsize=12)
    ax.set_xlim(-np.pi, np.pi)
    #ax.set_ylim(0, np.pi)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Event Display - Event {event_idx}', fontsize=14)
    
    fig.tight_layout()
    return fig

