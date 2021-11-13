import torch
import torch.nn as nn
import time
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from transformers import BertTokenizer


class IntervalLoss(nn.Module):
    def __init___(self):
        pass

    def forward(self, normed_y, label, interval_pred, alpha=0.1, soften = 80., lambda_in = 8.):
        import pdb; pdb.set_trace()
        y_pred_L, y_pred_U = interval_pred[:,0], interval_pred[:, 1]
        normed_ypos = normed_y[label == 1]
        normed_yneg = normed_y[label == 0]

        pred_label = torch.zeros_like(label)

        # positive samples loss 
        n = normed_ypos.shape[0]
        y_U_cap = y_pred_U > normed_ypos.reshape(-1)
        y_L_cap = y_pred_L < normed_ypos.reshape(-1)
        k_hard = y_U_cap*y_L_cap
        PICP = torch.sum(k_hard)/n
        
        MPIW_cap = torch.sum(k_hard * (y_pred_U - y_pred_L)) / (torch.sum(k_hard) + 0.001)
        pos_loss = MPIW_cap + lambda_in * np.sqrt(n) * (max(0,(1-alpha)-PICP)**2)

        # negative sample loss
        n = normed_yneg.shape[0]
        y_U_neg = y_pred_U > normed_yneg.reshape(-1)
        y_L_neg = y_pred_L < normed_yneg.reshape(-1)
        k_hard_neg = 1 - y_U_neg*y_L_neg

        # when upper bound is correct
        k_hard_neg = y_U_cap + y_L_cap
        PICP_neg = torch.sum(k_hard_neg)/n
        # in case didn't capture any need small no.
        MPIW_cap_neg = torch.sum(k_hard_neg * (y_pred_U - y_pred_L)) / (torch.sum(k_hard_neg) + 0.001)
        neg_loss = MPIW_cap_neg + lambda_in * np.sqrt(n) * (max(0,(1-alpha)-PICP_neg)**2)

        pred_label[label ==1] = k_hard
        pred_label[label ==0] = k_hard_neg
        
        return pos_loss + neg_loss, torch.sum(k_hard) + torch.sum(k_hard_neg), pred_label

class Multihead(nn.Module):
    def __init__(self, args, model):
        super(Multihead, self).__init__()

        self.model = model
        self.model.output_hidden_states = True
        self.binary_fc = nn.Linear(768 , 2)
        self.interval_fc = nn.Linear(768 , 2)

        self.cross_entropy = nn.CrossEntropyLoss(reduce='mean')
        self.interval_loss = IntervalLoss()

    def forward(self, input_ids, segment_ids, input_mask, label_ids, nor_val_s, heads):

        ypred = torch.zeros(heads.shape[0],2)
        pred_label = torch.zeros(heads.shape[0])

        loss = 0
        # import pdb; pdb.set_trace()
        qa_embed = self.model(input_ids, segment_ids, input_mask)
        
        
        cls_pred = self.binary_fc(qa_embed.logits[heads==1])
        cls_loss = self.cross_entropy(cls_pred, label_ids[heads==1])

        cls_pred_label = torch.argmax(cls_pred, axis=1)        
        cls_correct = (pred_label == label_ids).int().sum()

        interval_pred = self.interval_fc(qa_embed.logits[heads==2])
        interval_loss, interval_correct, interval_pred_label = self.interval_loss(nor_val_s[heads==2], label_ids[heads==2], interval_pred)

        ypred[head == 1] = cls_pred
        ypred[head == 2] = interval_pred

        pred_label[head ==1] = cls_pred_label
        pred_label[head ==2] = interval_pred_label
    
        return cls_loss + interval_loss, ypred, cls_correct + interval_correct, pred_label

