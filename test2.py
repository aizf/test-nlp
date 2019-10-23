from fast_bert.data_cls import BertDataBunch
import logging
import torch
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
device_cuda = torch.device("cuda")
# from transformers import BertTokenizer

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

DATA_PATH = r".\data\jigsaw"
LABEL_PATH = r".\data\jigsaw"
label_cols = [
    "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
]
databunch = BertDataBunch(
    DATA_PATH,
    LABEL_PATH,
    tokenizer='bert-base-uncased',
    train_file='train.csv',
    val_file='valid.csv',  # val.csv
    label_file='labels.csv',
    text_col='comment_text',
    label_col=label_cols,
    batch_size_per_gpu=16,
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
    output_dir=r".\data\jigsaw\output_dir",
    is_fp16=True,
    multi_gpu=True,
    multi_label=True)

learner.fit(4, lr=6e-5, schedule_type="warmup_linear")
learner.save_model()