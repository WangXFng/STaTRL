import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


from transformers import AlbertPreTrainedModel, AlbertModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class ALBertForClassification(AlbertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a CNN layer on top of
    the pooled output.
    """

    def __init__(self, config, num_labels=6):
        super(ALBertForClassification, self).__init__(config)
        self.num_labels = num_labels
        self.albert = AlbertModel.from_pretrained("albert-base-v2")

        # self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        self.fc=torch.nn.Linear(config.hidden_size,config.hidden_size)
        self.dropout=torch.nn.Dropout(0.8)
        self.classifier = nn.Linear(config.hidden_size, num_labels, bias=True)

    def forward(self, encode_dict, labels=None):
        # _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # print(encode_dict[0].size())
        # print(self.albert(**encode_dict))
        pooled_output = self.albert(**encode_dict)[1]

        # pooled_output = self.dense(pooled_output)
        # pooled_output = torch.relu(pooled_output)
        # pooled_output = F.normalize(pooled_output, p=2, dim=-1, eps=1e-05)
        logits=self.dropout(pooled_output)
        logits=self.fc(logits)
        logits=self.dropout(logits)
        logits=self.classifier(logits) #[batch_size,num_class]

        # # pooled_output = encode_dict[1]
        #
        # # print(pooled_output)
        # logits = self.classifier(pooled_output)
        #
        # logits = torch.softmax(logits, dim=-1)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # labels = F.one_hot(labels, self.num_labels)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits

    # def freeze_bert_encoder(self):
    #     for param in self.albert.parameters():
    #         param.requires_grad = False
    #
    # def unfreeze_bert_encoder(self):
    #     for param in self.albert.parameters():
    #         param.requires_grad = True

