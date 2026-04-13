# Storage of constants such as particle masses, PDG IDs, etc.

# PDG particle masses (MeV)
pdg_masses = {
    2212: 938.272,  # proton
    -2212: 938.272,  # anti-proton
    211: 139.570,  # pi+
    -211: 139.570,  # pi-
    321: 493.677,  # K+
    -321: 493.677,  # K-
    2112: 939.565,  # neutron (will be excluded)
    13: 105.658,  # mu-
    -13: 105.658,  # mu+
    111: 134.9766,  # pi0
    311: 497.6,  # K0
    -311: 497.6,  # K0
    11: 0.511,  # e-
    -11: 0.511,  # e+
    22: 0,  # photon
    130: 493.677,  # K0_L
    -2112: 939.565,  # anti-neutron
}

# PDG IDs: use kinetic energy (E - m) in available-energy sums
pdg_kinetic_energy = {2212, -2212}

# PDG IDs: use total energy (no mass subtraction), excluding muons
pdg_total_energy_no_muon = {22, 11, -11, 111, -111, 211, -211, 321, -321}

# PDG IDs: muons
pdg_muons = {13, -13}

# PDG IDs of commonly used particles
NEUTRAL_PION_PDG_ID = 111
PI_PLUS_PDG_ID = 211
PI_MINUS_PDG_ID = -211
