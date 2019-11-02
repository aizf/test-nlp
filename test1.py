import os
os.chdir("/content")
import torch
from transformers import PretrainedConfig, BertTokenizer, BertModel
from scripts.modeling import BertForMultiLabelSequenceClassification

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)
label_list = ["anger", "fear", "joy", "sadness", "surprise"]


def fine_tuning(batch, max_length=512):
    model.train()
    tokens_tensors = []
    attention_masks = []
    labels_tensors = []

    for text, labels in batch:
        # Tokenize input
        indexed_tokens = tokenizer.encode(text, add_special_tokens=True)
        seq_length = len(indexed_tokens)
        padding_length = max_length - seq_length  # 总输入长度 - 序列化的长度

        tokens_tensor = indexed_tokens + [0] * padding_length
        labels = [float(label) for label in labels]
        attention_mask = [1] * seq_length + [0] * padding_length

        tokens_tensors.append(tokens_tensor)
        attention_masks.append(attention_mask)
        labels_tensors.append(labels)

    # print(tokens_tensors)
    # print(labels_tensors)
    # If you have a GPU, put everything on cuda
    tokens_tensors = torch.tensor(tokens_tensors).to('cuda')
    attention_masks = torch.tensor(attention_masks).to('cuda')
    labels_tensors = torch.tensor(labels_tensors).to('cuda')
    # print(tokens_tensors)
    # print(labels_tensors)
    #
    with torch.no_grad():
        outputs = model(
            input_ids=tokens_tensors,
            attention_mask=attention_masks,
            labels=labels_tensors,
        )
        loss, logits = outputs[:2]
        print("loss: ", loss)
        print("logits: ", logits)


# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# model
config = {
    "attention_probs_dropout_prob": 0.1,
    "finetuning_task": None,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "num_labels": 5,
    "output_attentions": False,
    "output_hidden_states": False,
    "pruned_heads": {},
    "torchscript": False,
    "type_vocab_size": 2,
    "use_bfloat16": False,
    "vocab_size": 30522
}
pretrainedConfig = PretrainedConfig.from_dict(config)
model = BertForMultiLabelSequenceClassification.from_pretrained(
    'bert-base-uncased', config=pretrainedConfig)
# model.eval()  # Sets the module in evaluation mode(评估模式).
# To train the model, you should first set it back in training mode with model.train()
# If you have a GPU, put everything on cuda
model = model.to('cuda')
# # Parallelize the model architecture
# if self.multi_gpu == True:
#     self.model = torch.nn.DataParallel(self.model)

test_seq = "You let them take you, and that gave me enough time to escape."
test_label = [0, 0, 0, 0, 0]
test_seq1 = "We could just not go. - That's a bad idea. - We're already under attack."
test_label1 = [1, 1, 0, 0, 0]
batch = [(test_seq, test_label), (test_seq1, test_label1)]
fine_tuning(batch)
#
# with open(r"./data/trainset.txt") as f:
#     for line in f:
#         l = line.rstrip().split('##')
#         fine_tuning(l[0], l[1])
