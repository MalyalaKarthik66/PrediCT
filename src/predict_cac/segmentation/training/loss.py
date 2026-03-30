"""Loss configuration for heart segmentation.

DiceCELoss combines Dice (overlap) and CrossEntropy (class balance) terms,
which stabilizes training on the small, imbalanced foreground typical of heart
segmentation in COCA volumes.
"""
from __future__ import annotations

from typing import Dict

from monai.losses.dice import DiceCELoss
import torch


def get_loss_function(config: Dict) -> DiceCELoss:  # config kept for interface symmetry
    # Heavier weight on foreground to counter class imbalance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ce_weight = torch.tensor([0.1, 3.0], device=device)
    return DiceCELoss(to_onehot_y=True, softmax=True, weight=ce_weight)
