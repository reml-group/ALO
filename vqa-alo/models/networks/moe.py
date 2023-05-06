import torch
import torch.nn as nn
import torch.nn.functional as F
from block.models.networks.mlp import MLP
from .utils import grad_mul_const # mask_softmax, grad_reverse, grad_reverse_mask, 

eps = 1e-12

class MOE(nn.Module):

    def __init__(self, model, output_size, classif, end_classif=True):
        super().__init__()
        self.non_debias_net = model
        self.threshold = nn.Parameter(torch.rand(1))
        self.uniform = nn.Parameter(torch.rand(output_size))
        self.c_1 = MLP(**classif)
        self.end_classif = end_classif
        if self.end_classif:
            self.c_2 = nn.Linear(output_size, output_size)

    def forward(self, batch):
        out = {}
        # model prediction
        net_out = self.non_debias_net(batch)
        logits = net_out['logits']

        q_embedding = net_out['q_emb']  # N * q_emb
        q_embedding = grad_mul_const(q_embedding, 0.0)  # don't backpropagate through question encoder
        q_pred = self.c_1(q_embedding)

        dist = F.pairwise_distance(F.softmax(q_pred), self.uniform)
        gate = dist < self.threshold
        if gate:
            fusion_pred = logits
        else:
            fusion_pred = logits * torch.sigmoid(q_pred)

        q_out = None
        if self.end_classif and gate:
            q_out = self.c_2(q_pred)

        out['logits'] = fusion_pred
        out['logits_q'] = q_out
        out['gate'] = gate
        return out