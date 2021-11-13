import torch
import torch.nn as nn
import time
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from transformers import BertTokenizer
from bert_serving.client import BertClient

def np_QD_loss(y_true, y_pred_L, y_pred_U, alpha=0.1, soften = 80., lambda_in = 8.):
	"""
	manually (with np) calc the QD_hard loss
	"""
	n = y_true.shape[0]
	y_U_cap = y_pred_U > y_true.reshape(-1)
	y_L_cap = y_pred_L < y_true.reshape(-1)
	k_hard = y_U_cap*y_L_cap
	PICP = np.sum(k_hard)/n
	# in case didn't capture any need small no.
	MPIW_cap = np.sum(k_hard * (y_pred_U - y_pred_L)) / (np.sum(k_hard) + 0.001)
	loss = MPIW_cap + lambda_in * np.sqrt(n) * (max(0,(1-alpha)-PICP)**2)

	return loss

class Multihead(nn.Module):
    def __init__(self, args, model):
        super(Multihead, self).__init__()
        self.bert_client = BertClient()

        if args.orig_ckpt:
            self.orig_model = self.load_ckpt(model, args.orig_ckpt)
        else:
            self.orig_model = model

        
        self.interval_model =  model = BertForSequenceClassification.from_pretrained(args.interval_backbone,
                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))       
        self.interval_model.classifier = nn.Identity()
        self.option_fc = nn.Linear(768, 128)
        self.fc = nn.Linear(768 + 128 * 2, 2)
        self.sigmoid = nn.Sigmoid()
        if args.interval_ckpt: 
            self.interval_model = self.load_ckpt(self.interval_model, args.interval_ckpt)
        self.orig_model.output_hidden_states = True
        self.interval_model.output_hidden_states = True
  

    def load_ckpt(self, model, orig_ckpt):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.model.to(device)
            checkpoint = torch.load(orig_ckpt)
            try:
                self.model.load_state_dict(checkpoint['state_dict'])
            except:
                try:
                    self.model.module.load_state_dict(checkpoint['state_dict'])
                except:
                    self.model.load_state_dict(checkpoint)
        else:
            checkpoint = torch.load(orig_ckpt, map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'])



    def forward(self, input_ids, segment_ids, input_mask, label_ids, minmax_s, nor_val_s, heads):
        # sort out by head value
        # mask = torch.gather(head ==1)
        # 1,2 = input_ids[mask]
        if minmax_s:
            question_embed = self.interval_model(input_ids, segment_ids, input_mask)
                # (B, 768)
            minmax_embed = self.option_fc(minmax_s)
                # 
            embed = torch.cat([question_embed,minmax_embed], axis=1)
            output = self.fc(embed)
            output = self.sigmoid(output)
            # def np_QD_loss(y_true, y_pred_L, y_pred_U, alpha, soften = 80., lambda_in = 8.):
            # label_ids, nor_val_s
            # mask = label_ids ==1
            # loss = np_QD_loss(nor_val_s, mask, output[:,0], output[:,1])

        else:
            # original pipeline
            loss = self.orig_model(input_ids, segment_ids, input_mask, label_ids)

        return loss

