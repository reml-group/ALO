import torch.nn as nn
from bootstrap.lib.logger import Logger
from fractions import Fraction

class OrdinaryCriterion(nn.Module):

    def __init__(self, gamma = '1'):
        super().__init__()

        #assert '/' in gamma, "gamma must be written #/# format"
        self.gamma = float(Fraction(gamma))
        #assert 0.0 < self.gamma < 1.0
        Logger()(f'Ordinary loss (gamma={self.gamma})')
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, net_out, batch):
        out = {}
        logits = net_out['logits']
        class_id = batch['class_id'].squeeze(1)
        loss = self.gamma * self.criterion(logits, class_id)
        out['loss'] = loss
        return out
