import logging
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from BERT.pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertEmbeddings, BertEncoder

logger = logging.getLogger(__name__)

class BertModelNoPooler(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModelNoPooler, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        return encoded_layers

class BertForSequenceClassificationEncoder(BertPreTrainedModel):
    def __init__(self, config, output_all_encoded_layers=False, num_hidden_layers=None, fix_pooler=False):# 返回最后一层的表示sequence output
        super(BertForSequenceClassificationEncoder, self).__init__(config)
        if num_hidden_layers is not None:
            logger.info('num hidden layer is set as %d' % num_hidden_layers)
            config.num_hidden_layers = num_hidden_layers

        logger.info("Model config {}".format(config))
        if fix_pooler:
            self.bert = BertModelNoPooler(config)
        else:
            self.bert = BertModel(config)
        self.output_all_encoded_layers = output_all_encoded_layers
        self.apply(self.init_bert_weights)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        if self.output_all_encoded_layers:#False
            full_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
            return full_output, pooled_output
            # return  [full_output[i][:, 0] for i in range(len(full_output))], pool_output
        else:
            sequence_output,pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False) #False,仅输出最后一层的向量表示
            # return sequence_output, pooled_output
            trans_seq = sequence_output.transpose(0,1)
            assert len(trans_seq)==128
            return [trans_seq[i][...] for i in range(len(trans_seq))],pooled_output
'''
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        if self.output_all_encoded_layers:
            full_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
            return [full_output[i][:, 0] for i in range(len(full_output))], pooled_output
        else:
            _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
            return None, pooled_output
'''
class FCClassifierForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels, hidden_size, n_layers=0):
        super(FCClassifierForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.n_layers = n_layers
        for i in range(n_layers):
            setattr(self, 'fc%d' % i, nn.Linear(hidden_size, hidden_size))

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, encoded_feat):
        encoded_feat = self.dropout(encoded_feat)
        for i in range(self.n_layers):
            encoded_feat = getattr(self, 'fc%d' % i)(encoded_feat)
        logits = self.classifier(encoded_feat)
        return logits



