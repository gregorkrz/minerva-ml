FROM gkrz/devcontainer_omnilearn:1.0

RUN python -m pip install --no-cache-dir wandb
RUN python -m pip install --no-cache-dir transformers
