
# coding: UTF-8
import torch.nn as nn
from transformers import BertModel, BertConfig


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        model_config = BertConfig.from_pretrained(
           'bert-base-chinese', num_labels=config.num_classes)
        self.bert = BertModel.from_pretrained('bert-base-chinese',
                                              config=model_config)
    
        for param in self.bert.parameters():
            param.requires_grad = True
    
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[1]  
        token_type_ids = x[2]
        pooled = self.bert(context,
                              attention_mask=mask,
                              token_type_ids=token_type_ids).last_hidden_state[:, 0]
        out = self.fc(pooled)
        return out