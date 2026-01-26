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



## Event displays

The script will plot event displays with directions of the particles in the theta-phi plane. For blobs, we assume the direction from `(0, 0, 0)` to the primary vertex to the blob position.

```python -m src.scripts.make_event_displays --input_file <PATH_TO_ROOT_FILE> --output_dir <PATH_TO_OUTPUT_DIR> --n_events 10```


