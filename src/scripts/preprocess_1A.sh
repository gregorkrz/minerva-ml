
export DATA_DIR=/global/cfs/cdirs/m3246/gregork
#export DATA_DIR=/data
python -m src.scripts.preprocess_dataset --output-dir $DATA_DIR/Minerva/20260129 --max-workers 1 --max-workers-per-playlist 10 --playlists 1A
python -m src.scripts.split_dataset --input-dir $DATA_DIR/Minerva/20260129 --output-dir $DATA_DIR/Minerva/20260129_split --playlist 1A
python -m src.scripts.compute_enu_baselines --input-dir /pscratch/sd/g/gregork/MINERvA/raw_data/MediumEnergy_FHC_StandardMC_Playlist --output-dir $DATA_DIR/Minerva/20260129_split/baselines --playlist 1A