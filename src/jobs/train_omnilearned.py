# Train the transformer model locally
from src.jobs.slurm_template import SLURM_TEMPLATE_GPU
import os
from datetime import datetime as dt


training_cmd_PT_template = "python -m src.jobs.gen_train_cmds_OmniLearned -name SmallDataset_E_avail_LogMSE_PT_dscap{dscap}_Evts_seed_{dscapseed} --regress-E-available-no-muon -nw 10  --loss-type mse --log --use-pretrained pretrain_s --run --dataset-cap {dscap} --dataset-cap-seed {dscapseed}"
training_cmd_non_PT_template = "python -m src.jobs.gen_train_cmds_OmniLearned -name SmallDataset_E_avail_LogMSE_dscap{dscap}_Evts_seed_{dscapseed} --regress-E-available-no-muon -nw 10  --loss-type mse --log --run --dataset-cap {dscap} --dataset-cap-seed {dscapseed}"

for dscap in [100000, 500000, 1000000]:
    for dscapseed in [42]:
        training_cmd_PT = training_cmd_PT_template.format(dscap=dscap, dscapseed=dscapseed)
        training_cmd_non_PT = training_cmd_non_PT_template.format(dscap=dscap, dscapseed=dscapseed)
        print(training_cmd_PT)
        print(training_cmd_non_PT)
