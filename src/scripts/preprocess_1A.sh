
export DATA_DIR=/global/cfs/cdirs/m3246/gregork
#export DATA_DIR=/data
python -m src.scripts.preprocess_dataset --output-dir $DATA_DIR/Minerva/20260210_CCpi_labels --max-workers 1 --max-workers-per-playlist 10 --playlists 1A --use-max-blobs-and-prongs
python -m src.scripts.split_dataset --input-dir $DATA_DIR/Minerva/20260210_CCpi_labels --output-dir $DATA_DIR/Minerva/20260210_CCpi_labels_split --playlist 1A
python -m src.scripts.compute_enu_baselines --input-dir /pscratch/sd/g/gregork/MINERvA/raw_data/MediumEnergy_FHC_StandardMC_Playlist --output-dir $DATA_DIR/Minerva/20260210_CCpi_labels_split/baselines --playlist 1A

python -m src.scripts.preprocess_dataset --output-dir $DATA_DIR/Minerva/20260216_additional_info1 --max-workers 1 --max-workers-per-playlist 15 --playlists 1A --use-max-blobs-and-prongs
python -m src.scripts.split_dataset --input-dir $DATA_DIR/Minerva/20260216_additional_info1 --output-dir $DATA_DIR/Minerva/20260216_additional_info1_split --playlist 1A
python -m src.scripts.compute_enu_baselines --input-dir /pscratch/sd/g/gregork/MINERvA/raw_data/MediumEnergy_FHC_StandardMC_Playlist --output-dir $DATA_DIR/Minerva/20260216_additional_info1_split/baselines1 --playlist 1A

python -m src.scripts.preprocess_dataset --output-dir $DATA_DIR/Minerva/20260311 --max-workers 1 --max-workers-per-playlist 10  --use-max-blobs-and-prongs --max-blobs 20 --max-prongs 10 --playlists 1A 1B
python -m src.scripts.split_dataset --input-dir $DATA_DIR/Minerva/20260311 --output-dir $DATA_DIR/Minerva/20260311 --playlist 1A
python -m src.scripts.split_dataset --input-dir $DATA_DIR/Minerva/20260311 --output-dir $DATA_DIR/Minerva/20260311 --playlist 1B --val-ratio 0.005 --test-ratio 0.99
python -m src.scripts.compute_enu_baselines --input-dir /pscratch/sd/g/gregork/MINERvA/raw_data/MediumEnergy_FHC_StandardMC_Playlist --output-dir $DATA_DIR/Minerva/20260311/baselines --playlist 1A

python -m src.scripts.compute_enu_baselines --input-dir /pscratch/sd/g/gregork/MINERvA/raw_data/MediumEnergy_FHC_StandardMC_Playlist --output-dir $DATA_DIR/Minerva/20260311/baselines --playlist 1B


python -m src.scripts.preprocess_dataset --output-dir $DATA_DIR/Minerva/20260313 --max-workers 1 --max-workers-per-playlist 1  --use-max-blobs-and-prongs --max-blobs 20 --max-prongs 10 --playlists 1A 1B
python -m src.scripts.split_dataset --input-dir $DATA_DIR/Minerva/20260313 --output-dir $DATA_DIR/Minerva/20260313 --playlist 1A
python -m src.scripts.split_dataset --input-dir $DATA_DIR/Minerva/20260313 --output-dir $DATA_DIR/Minerva/20260313 --playlist 1B --val-ratio 0.005 --test-ratio 0.99

python -m src.scripts.extract_baselines --input-dir /pscratch/sd/g/gregork/MINERvA/raw_data/MediumEnergy_FHC_StandardMC_Playlist --output-dir $DATA_DIR/Minerva/20260311/baselines2 --playlist 1A
python -m src.scripts.extract_baselines --input-dir /pscratch/sd/g/gregork/MINERvA/raw_data/MediumEnergy_FHC_StandardMC_Playlist --output-dir $DATA_DIR/Minerva/20260313/baselines --playlist 1B
