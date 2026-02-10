from src.dataset.dataloader import load_data, HEPTorchDataset
from time import time

dataloader, class_weights = load_data("minerva_1A", "/global/cfs/cdirs/m3246/gregork/Minerva/20260201_all_max_blobs_and_prongs_split_fix_anomalies", batch=1024, dataset_type="train", mode="regression", distributed=False)
N_BATCHES = 10000
start_time = time()
i = 0
for batch in dataloader:
    if i % 1000 == 0:
        print("Batch ", i)
    if i >= N_BATCHES:
        break
    i += 1

print("Avg time per batch (regression): ", (time() - start_time) / N_BATCHES)


dataloader, class_weights = load_data("minerva_1A", "/global/cfs/cdirs/m3246/gregork/Minerva/20260201_all_max_blobs_and_prongs_split_fix_anomalies", batch=1024, dataset_type="train", mode="classification", distributed=False, classification_event_type=True)
N_BATCHES = 10000
start_time = time()
i = 0
for batch in dataloader:
    if i % 1000 == 0:
        print("Batch ", i)
    if i >= N_BATCHES:
        break
    i += 1

print("Avg time per batch (classification): ", (time() - start_time) / N_BATCHES)

