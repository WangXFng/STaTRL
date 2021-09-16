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

    def __init__(self, config, num_labels=2):
        super(ALBertForClassification, self).__init__(config)
        self.num_labels = num_labels
        self.albert = AlbertModel.from_pretrained("albert-base-v2")
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

        self.aspect_mlp = MLP(config.hidden_size, 3, 0.8)

        self.food_mlp = MLP(config.hidden_size, 2, 0.8)
        self.service_mlp = MLP(config.hidden_size, 2, 0.8)
        self.price_mlp = MLP(config.hidden_size, 2, 0.8)

        # self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        # self.label_fc = torch.nn.Linear(config.hidden_size,config.hidden_size)
        # self.dropout = torch.nn.Dropout(0.8)
        # self.label_classifier = nn.Linear(config.hidden_size, num_labels, bias=True)
        #
        # self.aspect_fc = torch.nn.Linear(config.hidden_size,config.hidden_size)
        # self.aspect_classifier = nn.Linear(config.hidden_size, num_labels, bias=True)

    # def forward(self, text, aspect=None, labels=None):
    def forward(self, text, aspect=None, polarity=None):

        encoded_input = self.tokenizer(text, return_tensors='pt', add_special_tokens=True, padding=True)
        device = torch.device('cuda')
        encoded_input = {key: tensor.to(device) for key, tensor in encoded_input.items()}
        albert_output = self.albert(**encoded_input)[1]
        # print(albert_output.size())

        aspect_res = self.aspect_mlp(albert_output)

        food_res = self.food_mlp(albert_output)
        price_res = self.price_mlp(albert_output)
        service_res = self.service_mlp(albert_output)

        category = torch.argmax(aspect_res)
        if category == 0:
            polarity_res = food_res
        elif category == 1:
            polarity_res = price_res
        else:
            polarity_res = service_res

        logist = torch.cat((food_res, price_res, service_res), dim=1)
        # print(logist)
        if aspect is not None:
            loss_fct = nn.CrossEntropyLoss()
            aspect_loss = loss_fct(aspect_res.view(-1, 3), aspect.view(-1))
            polarity_loss = loss_fct(polarity_res.view(-1, 2), polarity.view(-1))

            loss = aspect_loss + polarity_loss
            return loss, aspect_res, polarity_res
        else:
            return aspect_res, logist


class MLP(torch.nn.Module):
    def __init__(
            self,
            hidden_size, num_labels, droupout_rate):
        super().__init__()

        self.label_fc = torch.nn.Linear(hidden_size,hidden_size)
        self.dropout = torch.nn.Dropout(droupout_rate)
        self.label_classifier = nn.Linear(hidden_size, num_labels, bias=True)

    def forward(self, input):

        out = self.dropout(input)
        out = self.label_fc(out)
        out = self.dropout(out)
        out = self.label_classifier(out)

        return out