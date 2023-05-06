import torch.nn as nn
from bootstrap.lib.logger import Logger
from fractions import Fraction

class OrdinaryCriterion(nn.Module):

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, net_out, batch):
        out = {}
        logits = net_out['logits']
        class_id = batch['class_id'].squeeze(1)
        loss = self.criterion(logits, class_id)
        out['loss'] = loss
        return out
