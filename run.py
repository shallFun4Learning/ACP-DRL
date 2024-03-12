import pdb
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import TrainerCallback
import pandas as pd
import csv
import os
from transformers import BertForMaskedLM, BertTokenizer, pipeline, BertForSequenceClassification
from transformers import BertTokenizer, EarlyStoppingCallback
from transformers import TrainingArguments, Trainer
import datasets
from sklearn.model_selection import train_test_split
import Getmetrics
import getDataset
from transformers import BertModel, BertPreTrainedModel, BertForSequenceClassification
import torch.nn as nn
import torch
import torch.nn.functional as F
import argparse


parser = argparse.ArgumentParser(description='ACP-DRL')
parser.add_argument('--model_path', type=str,
                    help='path to the pre-trained model')
parser.add_argument('--tokenizer_path', type=str, help='path to the tokenizer')
parser.add_argument('--test_dataset_path', type=str,
                    help='path to the test dataset')
parser.add_argument('--do_lower', action='store_true',
                    help='whether to convert input text to lowercase')
parser.add_argument('--outPutDir', type=str, help='output directory')

args = parser.parse_args()


if not args.model_path:
    parser.error('model path is required')
model_path = args.model_path

tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.model_path

if not args.test_dataset_path:
    parser.error('test dataset path is required')

test_dataset_path = args.test_dataset_path
do_lower = args.do_lower
outPutDir = args.outPutDir if args.outPutDir else os.path.dirname(model_path)


class CustomBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_layers = 2
        self.bidirectional = True
        self.lstm = nn.LSTM(self.bert.config.hidden_size, self.hidden_size, num_layers=self.num_layers,
                            bidirectional=self.bidirectional, batch_first=True, dropout=0.2)
        self.classifier = nn.Linear(
            self.hidden_size * 2 if self.bidirectional else self.hidden_size, self.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        bert_output = outputs[0]
        seq_lens = attention_mask.sum(dim=1).cpu()
        seq_lens, perm_idx = seq_lens.sort(0, descending=True)
        bert_output = bert_output[perm_idx]
        packed_input = pack_padded_sequence(
            bert_output, seq_lens, batch_first=True)

        # LSTM
        packed_output, (hn, cn) = self.lstm(packed_input)
        forward_hidden_state = hn[-2] if self.bidirectional else hn[-1]
        backward_hidden_state = hn[-1]

        if self.bidirectional:
            hidden_states = torch.cat(
                (forward_hidden_state, backward_hidden_state), dim=1)
        else:
            hidden_states = forward_hidden_state
        _, unperm_idx = perm_idx.sort(0)
        hidden_states = hidden_states[unperm_idx]

        # 给分类器
        logits = self.classifier(hidden_states)

        if labels is not None:
            if self.num_labels == 1:
                # doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + (logits,)
        return outputs



tokenizer = BertTokenizer.from_pretrained(
    tokenizer_path, do_lower_case=do_lower)
print('Tokenizer have been loaded.')
model = CustomBertModel.from_pretrained(
    model_path)
print('BertModel have been loaded.')

model.config.max_position_embeddings = 512


test_dataset = getDataset.getDataset(modelpath=model_path, tkpath=tokenizer_path,
                                     datapath=test_dataset_path,
                                     max_len=model.config.max_position_embeddings)

test_args = TrainingArguments(
    output_dir=outPutDir,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    seed=1020,
    remove_unused_columns=True,
    dataloader_drop_last=False,
)
trainer = Trainer(
    model=model,
    args=test_args,
    compute_metrics=Getmetrics.getMetrics,
)
predictions = trainer.predict(test_dataset)
# pdb.set_trace()
res_dict = {
    'sequence': list(test_dataset['sequence']),
    'input_ids': list(test_dataset['input_ids']),
    'score': list(Getmetrics.getScore(predictions[0])[:, 1]),
    'predict_label': list(Getmetrics.getPredictLabel(predictions[0])),
    'y_label': list(test_dataset['label']),
}
res_df = pd.DataFrame(res_dict)
res_df.to_csv(outPutDir+os.sep+'predict_res.csv', index=None)
res_met_df = pd.DataFrame(
    {key: predictions[2][key] for key in list(predictions[2].keys())[1:-2]}, index=[1])
res_met_df = res_met_df.T
res_met_df = res_met_df.rename(columns={1: 'Model 1'})
res_met_df.to_csv(outPutDir+os.sep+'metrics_res.csv')
pd.set_option('display.float_format', '{:.10f}'.format)
print(res_met_df)
print(f'The results have been saved in {outPutDir}')
