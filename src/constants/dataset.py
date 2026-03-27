"""Dataset layout constants (must match preprocess_dataset / global_features)."""

# Base global cond width (before per-PID energy-sum columns): 3 log calorimetry + 3 log passive recoil
# + michel + 3 extras from compute_extra_global_features (reco muon, γγ mass, n charged-pion prongs)
GLOBAL_COND_BASE_DIM = 10
