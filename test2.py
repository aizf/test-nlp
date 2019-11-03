import os
# os.chdir("/content")
from fast_bert.data_cls import BertDataBunch
import logging
import torch

if not os.path.isdir(r"./output_dir/tensorboard"):
    os.makedirs(r"./output_dir/tensorboard")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
device_cuda = torch.device("cuda")
# from transformers import BertTokenizer

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

LABEL_PATH = DATA_PATH = r'./data'

label_cols = ["anger", "fear", "joy", "sadness", "surprise"]
databunch = BertDataBunch(
    DATA_PATH,
    LABEL_PATH,
    tokenizer='bert-base-uncased',
    train_file='train.csv',
    val_file='valid.csv',  # val.csv
    label_file='labels.csv',
    text_col='content',
    label_col=label_cols,
    batch_size_per_gpu=2,
    max_seq_length=512,
    multi_gpu=True,
    multi_label=True,
    model_type='bert')

from fast_bert.metrics import accuracy_multilabel
from fast_bert.learner_cls import BertLearner
metrics = [{'name': 'accuracy', 'function': accuracy_multilabel}]
learner = BertLearner.from_pretrained_model(
    databunch,
    pretrained_path='bert-base-uncased',
    metrics=metrics,
    device=device_cuda,
    logger=logger,
    output_dir=r"./output_dir",
    is_fp16=True,
    multi_gpu=True,
    multi_label=True)

learner.fit(
    6,
    lr=6e-5,
    validate=True,
    schedule_type="warmup_linear",
    optimizer_type="lamb")
learner.save_model()