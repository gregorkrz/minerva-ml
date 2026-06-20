FROM gkrz/devcontainer_omnilearn:1.0

COPY scripts/install_conda_ld_library_path_hooks.sh /tmp/install_conda_ld_library_path_hooks.sh
RUN bash /tmp/install_conda_ld_library_path_hooks.sh /opt/conda

RUN python -m pip install --no-cache-dir wandb
RUN python -m pip install --no-cache-dir transformers
