import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseCrossEntropyLoss(nn.Module):
    """
    CrossEntropy + sparsity penalty on feature attention.
    Compatible with criterion(outputs, targets).
    """

    def __init__(self, base_loss=None, lambda_sparse=1e-3):
        super().__init__()
        self.base_loss = base_loss or nn.CrossEntropyLoss(label_smoothing=0.1)
        self.lambda_sparse = lambda_sparse

    def forward(self, outputs, targets):
        """
        outputs can be:
        - logits
        - (logits, attention)
        """
        if isinstance(outputs, tuple):
            logits, attention = outputs
            sparsity_loss = attention.mean()
        else:
            logits = outputs
            sparsity_loss = 0.0

        ce_loss = self.base_loss(logits, targets)
        return ce_loss + self.lambda_sparse * sparsity_loss