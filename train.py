import os
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import TrainerCallback
import pandas as pd
import csv
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



class CustomBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config)
        self.num_layers = 2
        self.bidirectional = True
        self.dropout = nn.Dropout(0)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, self.hidden_size, num_layers=self.num_layers,
                            bidirectional=self.bidirectional, batch_first=True, dropout=0.25)
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


        logits = self.classifier(self.dropout(hidden_states))
        # logits = self.classifier(hidden_states)
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + (logits,)
        return outputs  # (loss), logits, (hidden_states), (attentions)


modelpath = '/data01/model'
do_lower = False
tokenizer = BertTokenizer.from_pretrained(
    modelpath, do_lower_case=do_lower)
print('Tokenizer have been loaded.')
model = CustomBertModel.from_pretrained(
    modelpath)
print('BertModel have been loaded.')

test_dataset = getDataset.getDataset(modelpath=modelpath, tkpath=modelpath,
                                     datapath="/data/ACP20AltTest.csv",
                                     max_len=2048)
dataset = datasets.load_dataset("csv", cache_dir='/data01/xf_bak/antiCancer/cache',
                                data_files="/data/ACP20AltTrain.csv")
dataset = dataset.shuffle(seed=702)
# 分词序列
tokenized_dataset = dataset.map(lambda x: tokenizer(
    ' '.join(x["sequence"]), max_length=2048,
    padding="max_length", truncation=True))

train_dataset = tokenized_dataset['train']

outPutDir = '/antiCancer/output' + \
    os.environ["CUDA_VISIBLE_DEVICES"]
training_args = TrainingArguments(
    output_dir=str(outPutDir),
    num_train_epochs=20,
    per_device_train_batch_size=4,
    warmup_ratio=0.1,
    weight_decay=0.01,
    learning_rate=(2e-5),
    logging_dir=str(outPutDir)+os.sep+'logs',
    logging_steps=50,
    save_strategy='epoch',  # 'epoch',
    save_total_limit=3,
    seed=1020,
    # data_seed=702,
    remove_unused_columns=True,
    lr_scheduler_type="cosine_with_restarts",
    optim='adafactor',  # 'adamw_torch',
    dataloader_drop_last=False,
)


trainer = Trainer(
    # the instantiated Transformers model to be trained
    model=model,
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,          # training dataset
    # the callback that computes metrics of interest
    compute_metrics=Getmetrics.getMetrics,
)

trainer.train()

predictions = trainer.predict(test_dataset)
res_dict = {
    'sequence': list(test_dataset['sequence']),
    'input_ids': list(test_dataset['input_ids']),
    'score': list(Getmetrics.getScore(predictions[0])[:, 1]),
    'predict_label': list(Getmetrics.getPredictLabel(predictions[0])),
    'y_label': list(predictions[1]),
    'check_label': list(test_dataset['label']),  # is from dataset
}


res_df = pd.DataFrame(res_dict)
res_df.to_csv(outPutDir+os.sep+'predict_res.csv', index=None)
res_met_df = pd.DataFrame(predictions[2], index=[1])
res_met_df = res_met_df.T
res_met_df = res_met_df.rename(columns={1: 'Model'})
res_met_df.to_csv(outPutDir+os.sep+'metrics_res.csv')
