import argparse
from src.dataset.event_displays import event_display
from src.preprocessing import get_event_collections
import uproot
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

np.random.seed(42)

human_readable_current = { # mc_current in the masterAnaDev
    1: "CC",
    2: "NC"
}

human_readable_intType = { # mc_intType in the masterAnaDev
    1: "CCQE",
    2: "Delta",
    3: "DIS",
    4: "Coherent",
    8: "2p2h"
}

parser = argparse.ArgumentParser(description='Make event displays')

# Make folders CC and NC, and then also for different int types
parser.add_argument('--input_file', type=str, required=True, help='Input file')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
parser.add_argument("--n_events", type=int, required=False, help='Max number of events to plot per interaction type', default=10)
args = parser.parse_args()

# Make folders CC and NC, and in each folder make folders for each intType that is present in the file for the given current

with uproot.open(args.input_file) as file:
    tree = file['MasterAnaDev']
    int_type = tree["mc_intType"].array().to_numpy()
    mc_current_type = tree["mc_current"].array().to_numpy()
    # get_event_collections
    mc_part, prong, blob, muons, photons = get_event_collections(tree)
    enu_mc = tree["mc_incomingE"].array().to_numpy()

    for key in human_readable_current.keys():
        os.makedirs(os.path.join(args.output_dir, human_readable_current[key]), exist_ok=True)
        for int_type_key in human_readable_intType.keys():
            # get events for the given current and int type
            available_idx = np.where((mc_current_type == key) & (int_type == int_type_key))[0]
            if len(available_idx) > 0:
                # shuffle and get the first n eventsž
                np.random.shuffle(available_idx)
                events = available_idx[:args.n_events]
                for event in events:
                    print("plotting event ", event, " of ", len(events), " for ", human_readable_current[key], " ", human_readable_intType[int_type_key])
                    mc_enu = enu_mc[event]
                    output_dir = Path(os.path.join(args.output_dir, human_readable_current[key], human_readable_intType[int_type_key]))
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_file = output_dir / f"event_{event}.pdf"
                    fig = event_display(event, muons=muons, photons=photons, mc_part=mc_part, blobs=blob, prongs=prong, title=f"{human_readable_current[key]} {human_readable_intType[int_type_key]} event {event} Enu_MC={mc_enu}")
                    fig.savefig(output_file)
                    #event_display_interactive(event, muons=muons, photons=photons, mc_part=mc_part, blobs=blob, prongs=prong, output_file=os.path.join(args.output_dir, human_readable_current[key], human_readable_intType[int_type_key], f"event_{event}.html"))
                    plt.close(fig)

