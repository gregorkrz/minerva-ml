"""Shared fixtures for the minerva-ml test suite."""

import numpy as np
import pytest
import torch
from pathlib import Path


@pytest.fixture(scope="session")
def synthetic_data_dir(tmp_path_factory):
    """Create a synthetic MINERvA dataset (jagged nested tensors) for integration tests."""
    base = tmp_path_factory.mktemp("minerva_data")
    data_dir = base / "1A"

    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        split_dir.mkdir(parents=True)

        n_events = 100 if split == "train" else 30

        events = []
        for _ in range(n_events):
            n_particles = np.random.randint(5, 34)
            event = torch.randn(n_particles, 10)
            event[:, 4] = torch.randint(0, 8, (n_particles,)).float()
            events.append(event)

        data = torch.nested.nested_tensor(events, layout=torch.jagged)

        truth_labels = torch.zeros(n_events, 15)
        truth_labels[:, 0] = torch.FloatTensor(n_events).uniform_(1000, 10000)
        int_types = torch.tensor([1, 2, 3, 4, 8])
        truth_labels[:, 1] = int_types[torch.randint(0, 5, (n_events,))].float()
        truth_labels[:, 3] = torch.tensor([1, 2])[
            torch.randint(0, 2, (n_events,))
        ].float()
        truth_labels[:, 5] = torch.randint(0, 3, (n_events,)).float()
        truth_labels[:, 6] = torch.randint(0, 3, (n_events,)).float()
        truth_labels[:, 8] = torch.FloatTensor(n_events).uniform_(0, 5000)
        truth_labels[:, 9] = torch.FloatTensor(n_events).uniform_(0, 4000)
        truth_labels[:, 10] = torch.randint(0, 3, (n_events,)).float()

        global_features = torch.randn(n_events, 16)

        torch.save(
            {
                "data": data,
                "truth_labels": truth_labels,
                "global_features": global_features,
            },
            split_dir / "0.pb",
        )

    return str(base)
