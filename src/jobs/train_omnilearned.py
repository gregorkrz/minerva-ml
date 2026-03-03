# Train the transformer model locally
from src.jobs.slurm_template import SLURM_TEMPLATE_GPU
import os
from datetime import datetime as dt

#OmniM_FullDataset_E_avail_Log1PLoss_PT_1A_20260301_231443
#FullDataset_E_avail_Log1PLoss_PT_1A_20260301_231300
#FullDataset_E_avail_Log1PLoss_1A_20260301_230852

training_cmd_PT_template = "python -m src.jobs.gen_train_cmds_OmniLearned -name FixBugSmallDataset_E_avail_Log1PLoss_PT_dscap{dscap}_Evts_seed_{dscapseed} --regress-E-available-no-muon -nw 10  --loss-type log1p --use-pretrained pretrain_s --run --dataset-cap {dscap} --dataset-cap-seed {dscapseed}"
training_cmd_non_PT_template = "python -m src.jobs.gen_train_cmds_OmniLearned -name FixBugSmallDataset_E_avail_Log1PLoss_dscap{dscap}_Evts_seed_{dscapseed} --regress-E-available-no-muon -nw 10  --loss-type log1p  --run --dataset-cap {dscap} --dataset-cap-seed {dscapseed}"

for dscap in [50000, 100000, 500000]:
    for dscapseed in [44]:
        training_cmd_PT = training_cmd_PT_template.format(dscap=dscap, dscapseed=dscapseed)
        training_cmd_non_PT = training_cmd_non_PT_template.format(dscap=dscap, dscapseed=dscapseed)
        print(training_cmd_PT)
        print(training_cmd_non_PT)



#training_cmd_PT_template = "python -m src.jobs.gen_train_cmds_OmniLearned -name FullDataset_E_avail_Log1PLoss_PT --regress-E-available-no-muon -nw 10  --loss-type log1p --use-pretrained pretrain_s --run "
#training_cmd_non_PT_template = "python -m src.jobs.gen_train_cmds_OmniLearned -name FullDataset_E_avail_Log1PLoss --regress-E-available-no-muon -nw 10  --loss-type log1p  --run "
#print(training_cmd_PT_template)
#print(training_cmd_non_PT_template)


#training_cmd_PT_template1 = "python -m src.jobs.gen_train_cmds_OmniLearned -name OmniM_FullDataset_E_avail_Log1PLoss_PT --regress-E-available-no-muon -nw 10  --loss-type log1p --use-pretrained pretrain_m --run "#print(training_cmd_PT_template1)