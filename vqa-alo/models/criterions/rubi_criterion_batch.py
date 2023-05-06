import torch.nn as nn
import torch
import torch.nn.functional as F
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options

class RUBiCriterionBatch(nn.Module):

    def __init__(self, question_loss_weight=1.0, loose_batch_num = '1'):
        super().__init__()

        Logger()(f'RUBiCriterion, with question_loss_weight = ({question_loss_weight})')

        self.question_loss_weight = question_loss_weight
        self.fusion_loss = nn.CrossEntropyLoss()
        self.question_loss = nn.CrossEntropyLoss()
        self.loose_batch_num = int(loose_batch_num)
        self.loss_tmp_list = []
        self.sum_list = [0, 0]
        self.gamma = 0.999
     
    def forward(self, net_out, batch):
        out = {}
        # logits = net_out['logits']
        logits_q = net_out['logits_q']
        logits_rubi = net_out['logits_all']
        class_id = batch['class_id'].squeeze(1)
        fusion_loss = self.fusion_loss(logits_rubi, class_id)
        self.loss_tmp_list.append(fusion_loss.item())  #!!!

        question_loss = self.question_loss(logits_q, class_id)
        loss = fusion_loss * self.gamma + self.question_loss_weight * question_loss

        if len(self.loss_tmp_list) < self.loose_batch_num:
            pass
        elif len(self.loss_tmp_list) == self.loose_batch_num:
            self.sum_list[-2] = self.sum_list[-1]
            self.sum_list[-1] = sum(self.loss_tmp_list)
            try:
                if self.sum_list[-2] / self.sum_list[-1] > 0:
                    self.gamma = min(self.sum_list[-2] / self.sum_list[-1], 0.999)
                else:
                    self.gamma = 0.999
            except:
                self.gamma = 0.999
            self.loss_tmp_list = []

        Logger()(f'RUBiCriterion, with gamma = {self.gamma}')
        out['loss'] = loss
        out['loss_mm_q'] = fusion_loss
        out['loss_q'] = question_loss
        return out

