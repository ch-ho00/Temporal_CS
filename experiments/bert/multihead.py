import torch
import torch.nn as nn
import time
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

class Multihead(nn.Module):
    def __init__(self, args, model):
        super(Multihead, self).__init__()
        self.orig_model = self.load_ckpt(model, args.orig_ckpt)
        self.interval_model =  model = BertForSequenceClassification.from_pretrained(args.interval_backbone,
                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))       

        self.orig_model.return_dict = True
        self.interval_model.return_dict = True 

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
        if minmax_s:
            loss, return_dict = self.interval_model(input_ids, segment_ids, input_mask, label_ids, minmax_s, nor_val_s, heads)
        else:
            # original pipeline
            loss, return_dict = self.orig_model(input_ids, segment_ids, input_mask, label_ids)

        return loss, return_dict

