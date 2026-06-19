"""Standard binary signal definitions for classification eval and training."""

from __future__ import annotations

# Pi_labels_v2 pid classes (-npi2 / get_Pi_labels_v2):
#   0 = CC with exactly one charged pion
#   1 = CC with more than one charged pion (CC-Npi)
#   2 = CC with one pi0 and no charged pions
#   3 = CC other (CC-Other)
#   4 = NC (not CC)
#
# Binary Weigh2 training (--binary-classifier --binary-signal CCN1pipm) maps
# pid {0, 1} -> label 1 (signal) and pid {2, 3, 4} -> label 0 (background).
# Eval npz ``pid`` keeps the original pid class for plotting.

CC1PI_CLASSES = [0]
CC1PI0_CLASSES = [2]
CCNPI_GE1_CLASSES = [0, 1]
CCNPI_GT1_CLASSES = [1]

# Internal keys used in metrics cache and plot scripts.
SIGNAL_TAG_TO_CLASSES: dict[str, list[int]] = {
    "cc1pi": CC1PI_CLASSES,
    "cc1pi0": CC1PI0_CLASSES,
    "ccnpi_ge1": CCNPI_GE1_CLASSES,
    "ccnpi_gt1": CCNPI_GT1_CLASSES,
}

# User-facing CLI / training aliases (case-insensitive lookup).
SIGNAL_ALIAS_TO_TAG: dict[str, str] = {
    "cc1pi": "cc1pi",
    "cc1pipm": "cc1pi",
    "cc1pi0": "cc1pi0",
    "ccnpi_ge1": "ccnpi_ge1",
    "ccn1pipm": "ccnpi_ge1",
    "ccnpi_gt1": "ccnpi_gt1",
    "ccnpipm": "ccnpi_gt1",
}

SIGNAL_TAG_CHOICES = sorted(SIGNAL_ALIAS_TO_TAG.keys())


def resolve_signal_classes(signal_tag: str) -> list[int]:
    """Resolve a user-facing signal tag to the list of ``pid`` / training classes."""
    key = signal_tag.strip().lower()
    if key not in SIGNAL_ALIAS_TO_TAG:
        allowed = ", ".join(SIGNAL_TAG_CHOICES)
        raise ValueError(
            f"Unknown signal tag '{signal_tag}'. Allowed values: {allowed}"
        )
    internal = SIGNAL_ALIAS_TO_TAG[key]
    return list(SIGNAL_TAG_TO_CLASSES[internal])
