from src.dataset.dataloader import load_data, HEPTorchDataset, Task
from time import time

dataloader, class_weights = load_data("minerva_1A", "/global/cfs/cdirs/m3246/gregork/Minerva/20260216_additional_info_split", batch=2048, dataset_type="train", distributed=True, task=Task(type="regression"), num_workers=50)
N_BATCHES = 1000
start_time = time()
i = 0
print("Starting to iterate")
for batch in dataloader:
    if i % 100 == 0:
        print("Batch ", i)
    if i >= N_BATCHES:
        break
    i += 1

print("total time , for {} batches: {} seconds".format(N_BATCHES, time() - start_time))
print("Avg time per batch (classification): ", (time() - start_time) / N_BATCHES)

