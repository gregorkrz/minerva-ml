export DATA_DIR=/global/cfs/cdirs/m3246/gregork
#export DATA_DIR=/data

python -m src.scripts.preprocess_dataset --output-dir $DATA_DIR/Minerva/20260129_all --max-workers 1 --max-workers-per-playlist 10 
python -m src.scripts.split_dataset --input-dir $DATA_DIR/Minerva/20260129_all --output-dir $DATA_DIR/Minerva/20260129_split_all
python -m src.scripts.compute_enu_baselines --input-dir /pscratch/sd/g/gregork/MINERvA/raw_data/MediumEnergy_FHC_StandardMC_Playlist --output-dir $DATA_DIR/Minerva/20260129_split_all/baselines


python -m src.scripts.preprocess_dataset --output-dir $DATA_DIR/Minerva/20260201_all_max_blobs_and_prongs --max-workers 1 --max-workers-per-playlist 10 --use-max-blobs-and-prongs --playlists 1A
python -m src.scripts.split_dataset --input-dir $DATA_DIR/Minerva/20260201_all_max_blobs_and_prongs --output-dir $DATA_DIR/Minerva/20260201_all_max_blobs_and_prongs_split_fix_anomalies --playlist 1A
python -m src.scripts.compute_enu_baselines --input-dir /pscratch/sd/g/gregork/MINERvA/raw_data/MediumEnergy_FHC_StandardMC_Playlist --output-dir $DATA_DIR/Minerva/20260201_all_max_blobs_and_prongs/baselines

export DATA_DIR=/global/cfs/cdirs/m3246/gregork

python -m src.jobs.train --dataset Minerva_v2 --training-name "max_blob_and_prong_log" --playlist 1A --loss mse --max-particles 33 --print-cmd-only --regress-log -bs 512 -nw 64
python -m src.jobs.train --dataset Minerva_v2 --training-name "max_blob_and_prong" --playlist 1A --loss l1 --max-particles 33 --print-cmd-only

# 20260201_all_max_blobs_and_prongs_split_fix_anomalies