import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize input
text = "Hello, my dog is cute"
indexed_tokens = tokenizer.convert_tokens_to_ids(
    ['[CLS]'] + tokenizer.tokenize(text) + ['[SEP]'])
# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor(indexed_tokens).unsqueeze(0)
# 相当于 tokens_tensor = torch.tensor([indexed_tokens])

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1

# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to('cuda')
model = model.to('cuda')
labels = labels.to('cuda')

#
with torch.no_grad():
    outputs = model(tokens_tensor, labels=labels)
    loss, logits = outputs[:2]

print(tokens_tensor)
print(outputs)
print(loss)
print(logits)
