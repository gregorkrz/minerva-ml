# minerva-data-processing

This folder contains lots of duplicate code! Probably the latest notebook version will contain cleanest and most useful info.
Work in progress.

## Download the data

First, make sure that the `SCRATCH` environment variable is set.

* To download MC: ```python -m src.scripts.download_data```
* To download data: ```python -m src.scripts.download_data --prefix MediumEnergy_FHC_Data_Playlist```

## Preprocess the data

```python -m src.scripts.preprocess_dataset --output-dir <OUTPUT_DIR>```

This doesn't do any filtering.

## Data selection and split

```python -m src.scripts.split_dataset --input-dir /data/Minerva/20260127_nested  --output-dir /data/Minerva/20260127_nested_split```

To investigate the features of the created dataset, look at `notebooks/stats.ipynb`.

## Event displays

The script will plot event displays with directions of the particles in the theta-phi plane. For blobs, we assume the direction from `(0, 0, 0)` to the primary vertex to the blob position.

```python -m src.scripts.make_event_displays --input_file <PATH_TO_ROOT_FILE> --output_dir <PATH_TO_OUTPUT_DIR> --n_events 10```

## Training (OmniLearned repo)

`python -m omnilearned.cli train --dataset minerva_1A --path /data/Minerva/20260127_nested_split --output_dir ./test_run --save_tag "test" --size small --num_feat 4 --use_pid --pid_idx 4 --pid_dim 6 --conditional --num_cond 4 --mode regression_E_nu --num_classes 1 --batch 16 --epoch 2 --lr 5e-5 --num_workers 2 --nevts 1000`

