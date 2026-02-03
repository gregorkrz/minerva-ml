
export DATA_DIR=/global/cfs/cdirs/m3246/gregork


# Classification
python -m src.jobs.train --dataset Minerva_v2 --training-name "classification_event_type" --playlist 1A --class-event-type --max-particles 33 --print-cmd-only -bs 1024 -nw 8
python -m src.jobs.train --dataset Minerva_v2 --training-name "classification_current_type" --playlist 1A --class-current-type --max-particles 33 --print-cmd-only -bs 1024 -nw 8

# Classification with pretrained models
python -m src.jobs.train --dataset Minerva_v2 --training-name "classification_event_type_Pretrained" --playlist 1A --class-event-type --max-particles 33 --print-cmd-only -bs 1024 -nw 8 --use-pretrained pretrain_s
python -m src.jobs.train --dataset Minerva_v2 --training-name "classification_current_type_Pretrained" --playlist 1A --class-current-type --max-particles 33 --print-cmd-only -bs 1024 -nw 8 --use-pretrained pretrain_s


# Using the log loss - cont. 
python -m src.jobs.train --dataset Minerva_v2 --training-name "CONT_max_blob_and_prong_log" --playlist 1A --loss mse --max-particles 33 --print-cmd-only --regress-log -bs 1024 -nw 2 --use-pretrained  /global/cfs/cdirs/m3246/gregork/checkpoints/max_blob_and_prong_log_1A_20260203_023142/best_model_max_blob_and_prong_log_1A_20260203_023142.pt
python -m src.jobs.train --dataset Minerva_v2 --training-name "CONT_max_blob_and_prong_log_PretrainedSmall" --playlist 1A --loss mse --max-particles 33 --print-cmd-only --regress-log -bs 1024 -nw 2 --use-pretrained  /global/cfs/cdirs/m3246/gregork/checkpoints/max_blob_and_prong_log_PretrainedSmall_1A_20260203_025039/best_model_max_blob_and_prong_log_PretrainedSmall_1A_20260203_025039.pt



# Using the log loss
python -m src.jobs.train --dataset Minerva_v2 --training-name "max_blob_and_prong_log" --playlist 1A --loss mse --max-particles 33 --print-cmd-only --regress-log -bs 1024 -nw 8
python -m src.jobs.train --dataset Minerva_v2 --training-name "max_blob_and_prong_log_PretrainedSmall" --playlist 1A --loss mse --max-particles 33 --print-cmd-only --regress-log -bs 1024 -nw 8 --use-pretrained pretrain_s


# Without the log, it seems like the loss is dominated by the large anomalies
python -m src.jobs.train --dataset Minerva_v2 --training-name "max_blob_and_prong" --playlist 1A --loss l1 --max-particles 33 --print-cmd-only

# 20260201_all_max_blobs_and_prongs_split_fix_anomalies

