
export DATA_DIR=/global/cfs/cdirs/m3246/gregork
#export DATA_DIR=/data
python -m src.scripts.preprocess_dataset --output-dir $DATA_DIR/Minerva/20260210_CCpi_labels --max-workers 1 --max-workers-per-playlist 10 --playlists 1A --use-max-blobs-and-prongs
python -m src.scripts.split_dataset --input-dir $DATA_DIR/Minerva/20260210_CCpi_labels --output-dir $DATA_DIR/Minerva/20260210_CCpi_labels_split --playlist 1A
python -m src.scripts.compute_enu_baselines --input-dir /pscratch/sd/g/gregork/MINERvA/raw_data/MediumEnergy_FHC_StandardMC_Playlist --output-dir $DATA_DIR/Minerva/20260210_CCpi_labels_split/baselines --playlist 1A

python -m src.scripts.preprocess_dataset --output-dir $DATA_DIR/Minerva/20260216_additional_info1 --max-workers 1 --max-workers-per-playlist 15 --playlists 1A --use-max-blobs-and-prongs
python -m src.scripts.split_dataset --input-dir $DATA_DIR/Minerva/20260216_additional_info1 --output-dir $DATA_DIR/Minerva/20260216_additional_info1_split --playlist 1A
python -m src.scripts.compute_enu_baselines --input-dir /pscratch/sd/g/gregork/MINERvA/raw_data/MediumEnergy_FHC_StandardMC_Playlist --output-dir $DATA_DIR/Minerva/20260216_additional_info1_split/baselines1 --playlist 1A



python -m src.scripts.preprocess_dataset --output-dir $DATA_DIR/Minerva/20260223_150obj --max-workers 1 --max-workers-per-playlist 15 --playlists 1A --max-objects 150
python -m src.scripts.split_dataset --input-dir $DATA_DIR/Minerva/20260223_150obj --output-dir $DATA_DIR/Minerva/20260223_150obj_split --playlist 1A
python -m src.scripts.compute_enu_baselines --input-dir /pscratch/sd/g/gregork/MINERvA/raw_data/MediumEnergy_FHC_StandardMC_Playlist --output-dir $DATA_DIR/Minerva/20260223_150obj/baselines --playlist 1A



# /global/cfs/cdirs/m3246/gregork/Minerva/20260216_additional_info1_split

