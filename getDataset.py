import Getmetrics
from sklearn.model_selection import train_test_split
import datasets
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer
from transformers import BertForMaskedLM, BertTokenizer, pipeline, BertForSequenceClassification
import os

def getDataset(modelpath='', tkpath='', datapath='', max_len=512):
    do_lower = False
    tokenizer = BertTokenizer.from_pretrained(
        tkpath, do_lower_case=do_lower)
    print('Tokenizer have been loaded.')
    dataset = datasets.load_dataset("csv", cache_dir='/data01/xf_bak/antiCancer/cache/',
                                    data_files=datapath)
    dataset = dataset.shuffle(seed=702)
    print('Dataset have been loaded.')
    tokenized_dataset = dataset.map(lambda x: tokenizer(
        ' '.join(x["sequence"]), max_length=max_len,
        padding="max_length", truncation=True))
    return tokenized_dataset['train']
