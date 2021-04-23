import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


from transformers import AlbertPreTrainedModel, AlbertModel, AlbertTokenizer
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
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

        # self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        self.label_fc = torch.nn.Linear(config.hidden_size,config.hidden_size)
        self.dropout = torch.nn.Dropout(0.8)
        self.label_classifier = nn.Linear(config.hidden_size, num_labels, bias=True)

        self.aspect_fc = torch.nn.Linear(config.hidden_size,config.hidden_size)
        self.aspect_classifier = nn.Linear(config.hidden_size, num_labels, bias=True)

    # def forward(self, text, aspect=None, labels=None):
    def forward(self, text, labels=None):
        # _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # print(encode_dict[0].size())
        # print(self.albert(**encode_dict))

        encoded_input = self.tokenizer(text, return_tensors='pt', add_special_tokens=True, padding=True)
        device = torch.device('cuda')
        encoded_input = {key: tensor.to(device) for key, tensor in encoded_input.items()}
        pooled_output = self.albert(**encoded_input)[1]

        # pooled_output = self.dense(pooled_output)
        # pooled_output = torch.relu(pooled_output)
        # pooled_output = F.normalize(pooled_output, p=2, dim=-1, eps=1e-05)
        pooled_output = self.dropout(pooled_output)

        # aspect_logits = self.aspect_fc(pooled_output)
        # aspect_logits = self.dropout(aspect_logits)
        # aspect_logits = self.aspect_classifier(aspect_logits)
        # aspect_logits = torch.sigmoid(aspect_logits)

        label_logits = self.label_fc(pooled_output)
        label_logits = self.dropout(label_logits)
        label_logits = self.label_classifier(label_logits)
        # label_logits = F.softmax(label_logits, dim=-1)
        # label_logits = torch.sigmoid(label_logits)

        # # pooled_output = encode_dict[1]

        # if labels is not None and aspect is not None :
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # labels = F.one_hot(labels, self.num_labels)

            labels_loss = loss_fct(label_logits.view(-1, 6), labels.view(-1))
            # aspect_loss = loss_fct(aspect_logits.view(-1, self.num_labels), aspect.view(-1))

            loss = labels_loss
            # loss = labels_loss + aspect_loss
            # return loss, aspect_logits, label_logits
            return loss, label_logits
        else:
            # return aspect_logits, label_logits
            return label_logits

    # def freeze_bert_encoder(self):
    #     for param in self.albert.parameters():
    #         param.requires_grad = False
    #
    # def unfreeze_bert_encoder(self):
    #     for param in self.albert.parameters():
    #         param.requires_grad = True

