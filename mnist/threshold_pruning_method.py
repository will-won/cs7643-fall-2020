import torch
from torch.nn.utils import prune


class ThresholdPruningMethod(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold: float):
        super(ThresholdPruningMethod, self).__init__()

        self.threshold = threshold

    def compute_mask(self, t, default_mask):
        return torch.abs(t) > self.threshold
