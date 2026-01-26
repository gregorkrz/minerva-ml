# minerva-data-processing

This folder contains lots of duplicate code! Probably the latest notebook version will contain cleanest and most useful info.
Work in progress.

## Download the data
First, make sure that the `SCRATCH` environment variable is set.

* To download MC: ```python -m src.download_data```
* To download data: ```python -m src.download_data --prefix MediumEnergy_FHC_Data_Playlist```

## Preprocess the data
```python -m src.preprocess_dataset --output-dir <OUTPUT_DIR>```

This doesn't do any filtering.

## Data selection and split

## Event displays

```python -m src.make_event_displays  ```