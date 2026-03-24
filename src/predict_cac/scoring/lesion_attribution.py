"""Assign CAC lesions to vessel-zone masks."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


VESSELS = ("LAD", "LCX", "RCA", "LM")


def attribute_lesions_to_vessels(
    calcium_mask: np.ndarray,
    vessel_zone_masks: dict[str, np.ndarray],
) -> dict[str, int]:
    """Count connected calcium lesions per vessel territory."""
    labeled, n = ndimage.label(calcium_mask > 0)
    counts = {v: 0 for v in VESSELS}

    for label_id in range(1, n + 1):
        lesion = labeled == label_id
        best_vessel = None
        best_overlap = -1
        for vessel, zone in vessel_zone_masks.items():
            overlap = int(np.logical_and(lesion, zone > 0).sum())
            if overlap > best_overlap:
                best_overlap = overlap
                best_vessel = vessel
        if best_vessel is not None and best_overlap > 0:
            counts[best_vessel] = counts.get(best_vessel, 0) + 1

    return counts
