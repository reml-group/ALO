import torch.nn as nn
import torch
from bootstrap.lib.logger import Logger
from fractions import Fraction

class OrdinaryCriterionBatch(nn.Module):

    def __init__(self, loose_batch_num):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss()
        self.loose_batch_num = int(loose_batch_num)
        self.loss_tmp_list = []
        self.sum_list = [0, 0]
        self.gamma = 0.999
        Logger()(f'batch number = {self.loose_batch_num}')

    def forward(self, net_out, batch):
        #========= original ============
        # out = {}
        # logits = net_out['logits']
        # class_id = batch['class_id'].squeeze(1)
        # loss = self.gamma * self.criterion(logits, class_id)
        # out['loss'] = loss
        # return out

        #========== batch=1 ============
        # out = {}
        # logits = net_out['logits']
        # class_id = batch['class_id'].squeeze(1)
        # loss_tmp = self.criterion(logits, class_id)
        # self.lose_list.append(loss_tmp)
        # if self.last_loss < 0:
        #     gamma = 0.999
        #     self.last_loss = loss_tmp.detach().item()
        # else:
        #     tmp = loss_tmp.detach().item()
        #     gamma = min(self.last_loss/tmp, 0.999)
        #     self.last_loss = tmp
        # Logger()(f'(gamma={gamma})')
        # loss = gamma * loss_tmp
        # out['loss'] = loss

        #============ batch = self.loose_batch_num ==========
        out = {}
        logits = net_out['logits']
        class_id = batch['class_id'].squeeze(1)
        loss_tmp = self.criterion(logits, class_id)

        self.loss_tmp_list.append(loss_tmp.item())  #!!!
        # Logger()(f'(list={[a.item() for a in self.loss_tmp_list]})')
        loss = self.gamma * loss_tmp
        # loss.detach()
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
                
        Logger()(f'(gamma={self.gamma})')
        
        out['loss'] = loss
        return out
