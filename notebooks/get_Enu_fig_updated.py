def get_Enu_fig(mc_E, reco_e_dict: dict):
    # reco_e_dict: key should be what's plotted. value should be same shape as mc_E, if it's not reconstructed, it should have value -1.
    # fig should have 2 x 2 subplots. left: counts; right: normalized histograms to unit area.
    # Top: E / 1000 histograms (GeV); Bottom: E_reco/E_mc
    import numpy as np
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    bins_E = np.linspace(0, 20, 200)
    bins_E_reco_over_true = np.linspace(0, 2, 100)
    ax[0, 0].hist(mc_E, bins=bins_E, histtype='step', color='black', label='MC E')
    ax[0, 1].hist(mc_E, bins=bins_E, histtype='step', color='black', label='MC E', density=True)
    
    # Store statistics for text boxes
    stats_text_10 = []  # for ax[1,0]
    stats_text_11 = []  # for ax[1,1]
    
    for method in reco_e_dict.keys():
        mask = reco_e_dict[method] > 0
        percentage_reconstructed = round(np.sum(mask) / len(mask) * 100, 1)
        ax[0, 0].hist(reco_e_dict[method], bins=bins_E, histtype='step', label=method + f" ({percentage_reconstructed}%)")
        ax[0, 1].hist(reco_e_dict[method], bins=bins_E, histtype='step', label=method + f" ({percentage_reconstructed}%)", density=True)
        E_reco_over_true = reco_e_dict[method][mask] / mc_E[mask]
        
        # MPV (Most Probable Value) - use the bin with max count
        counts, bin_edges = np.histogram(E_reco_over_true, bins=bins_E_reco_over_true)
        mpv_idx = np.argmax(counts)
        mpv = (bin_edges[mpv_idx] + bin_edges[mpv_idx + 1]) / 2
        
        # Calculate RMS from MPV
        rms = np.sqrt(np.mean((E_reco_over_true - mpv)**2))
        
        ax[1, 0].hist(E_reco_over_true, bins=bins_E_reco_over_true, histtype='step', label=method + f" ({percentage_reconstructed}%)")
        ax[1, 1].hist(E_reco_over_true, bins=bins_E_reco_over_true, histtype='step', label=method + f" ({percentage_reconstructed}%)", density=True)
        
        # Add statistics to text lists
        stats_text_10.append(f"{method}: RMS={rms:.3f}, MPV={mpv:.3f}")
        stats_text_11.append(f"{method}: RMS={rms:.3f}, MPV={mpv:.3f}")
    
    ax[0, 0].legend()
    ax[0, 0].set_xlabel("Energy [GeV]")
    ax[0, 0].set_ylabel("Counts")
    ax[0, 0].set_title("E nu Distribution")
    ax[0, 1].legend()
    ax[0, 1].set_xlabel("Energy [GeV]")
    ax[0, 1].set_ylabel("Relative counts")
    ax[0, 1].set_title("E nu Distribution (normalized)")
    ax[1, 0].legend()
    ax[1, 0].set_xlabel("E reco / E true")
    ax[1, 0].set_ylabel("Counts")
    ax[1, 0].set_title("E reco / E true Distribution")
    ax[1, 1].legend()
    ax[1, 1].set_xlabel("E reco / E true")
    ax[1, 1].set_ylabel("Relative counts")
    ax[1, 1].set_title("E reco / E true Distribution (normalized)")
    
    # Add statistics text boxes to bottom plots
    stats_text_str_10 = '\n'.join(stats_text_10)
    stats_text_str_11 = '\n'.join(stats_text_11)
    
    ax[1, 0].text(0.98, 0.97, stats_text_str_10, transform=ax[1, 0].transAxes,
                  fontsize=9, verticalalignment='top', horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax[1, 1].text(0.98, 0.97, stats_text_str_11, transform=ax[1, 1].transAxes,
                  fontsize=9, verticalalignment='top', horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Put a grid on all
    ax[0, 0].grid(True)
    ax[0, 1].grid(True)
    ax[1, 0].grid(True)
    ax[1, 1].grid(True)

    # put legends on upper left for bottom plots to avoid overlap with stats
    ax[0, 0].legend(loc='lower right')
    ax[0, 1].legend(loc='lower right')
    ax[1, 0].legend(loc='upper left')
    ax[1, 1].legend(loc='upper left')
    fig.tight_layout()
    return fig
