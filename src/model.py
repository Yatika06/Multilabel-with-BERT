import torch
from transformers import BertModel


class BERTClassifier(torch.nn.Module):
    def __init__(self, drop_out):
        super().__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=False)
        self.drop_out = drop_out
        self.dropout = torch.nn.Dropout(self.drop_out)
        self.linear = torch.nn.Linear(768, 6)

    def forward(self, input_ids, attn_mask, token_type_ids):
        _, output_1 = self.bert_model(
            input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output_1)
        output = self.linear(output_dropout)
        return output
