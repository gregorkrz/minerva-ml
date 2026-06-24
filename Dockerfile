FROM gkrz/devcontainer_omnilearn:1.0

# Prefer conda's libstdc++ so scipy/sklearn don't hit "CXXABI_1.3.15 not found".
ENV LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH}

RUN python -m pip install --no-cache-dir wandb
RUN python -m pip install --no-cache-dir transformers
