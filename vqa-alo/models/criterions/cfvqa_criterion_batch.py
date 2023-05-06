import torch.nn as nn
import torch
import torch.nn.functional as F
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options

class CFVQACriterionBatch(nn.Module):

    def __init__(self, question_loss_weight=1.0, vision_loss_weight=1.0, is_va=True, loose_batch_num = '1'):
        super().__init__()
        self.is_va = is_va

        Logger()(f'CFVQACriterion, with question_loss_weight = ({question_loss_weight})')
        if self.is_va:
            Logger()(f'CFVQACriterion, with vision_loss_weight = ({vision_loss_weight})')

        self.fusion_loss = nn.CrossEntropyLoss()
        self.question_loss = nn.CrossEntropyLoss()
        self.question_loss_weight = question_loss_weight
        if self.is_va:
            self.vision_loss = nn.CrossEntropyLoss()
            self.vision_loss_weight = vision_loss_weight
        self.loose_batch_num = int(loose_batch_num)
        self.loss_tmp_list = []
        self.sum_list = [0, 0]
        self.gamma = 0.999
        
    def forward(self, net_out, batch):
        out = {}
        class_id = batch['class_id'].squeeze(1)
        
        logits_rubi = net_out['logits_all']
        fusion_loss = self.fusion_loss(logits_rubi, class_id)

        # ============batch=============
        # fusion_loss = self.fusion_loss(logits_rubi, class_id)
        # if self.last_loss < 0:
        #     gamma = 0.999
        #     self.last_loss = fusion_loss.detach().item()
        # else:
        #     tmp = fusion_loss.detach().item()
        #     gamma = min(self.last_loss/tmp, 0.999)
        #     self.last_loss = tmp
        # ===============================
        self.loss_tmp_list.append(fusion_loss.item())  #!!!
        
        logits_q = net_out['logits_q']
        question_loss = self.question_loss(logits_q, class_id)

        if self.is_va:
            logits_v = net_out['logits_v']
            vision_loss = self.vision_loss(logits_v, class_id)

        nde = net_out['z_nde']
        p_te = torch.nn.functional.softmax(logits_rubi, -1).clone().detach()
        p_nde = torch.nn.functional.softmax(nde, -1)
        kl_loss = - p_te*p_nde.log()    
        kl_loss = kl_loss.sum(1).mean() 

        # fusion_loss * gamma
        loss = fusion_loss * self.gamma \
                + self.question_loss_weight * question_loss \
                + kl_loss
        if self.is_va:
            loss += self.vision_loss_weight * vision_loss

        # ===============
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
            Logger()(f'(sum_list={self.sum_list})')
        
        Logger()(f'gamma = ({self.gamma})')
        out['loss'] = loss
        out['loss_mm_q'] = fusion_loss
        out['loss_q'] = question_loss
        if self.is_va:
            out['loss_v'] = vision_loss
        return out
