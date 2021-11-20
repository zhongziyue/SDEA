from transformers import BertModel
import torch as t
from config.KBConfig import *


class BasicBertModel(t.nn.Module):
    def __init__(self, pretrain_bert_path):
        super(BasicBertModel, self).__init__()
        bert_config = BertConfig.from_pretrained(pretrain_bert_path)
        self.bert_model = BertModel.from_pretrained(pretrain_bert_path, config=bert_config)
        self.out_linear_layer = t.nn.Linear(bert_config.hidden_size, bert_output_dim)
        self.dropout = t.nn.Dropout(p=0.1)

    def forward(self, tids, masks):
        bert_out = self.bert_model(input_ids=tids, attention_mask=masks)
        last_hidden_state = bert_out.last_hidden_state
        cls = last_hidden_state[:, 0]
        output = self.dropout(cls)
        output = self.out_linear_layer(output)
        # output = t.nn.functional.normalize(output, p=2, dim=-1)
        return output
