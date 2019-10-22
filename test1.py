import torch
from transformers import PretrainedConfig, BertTokenizer, BertModel, BertForSequenceClassification

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)


def fine_tuning(text, label):
    # Tokenize input
    text = text
    indexed_tokens = tokenizer.encode(text, add_special_tokens=True)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor(indexed_tokens).unsqueeze(0)
    # 相当于 tokens_tensor = torch.tensor([indexed_tokens])

    # label
    label_list = ["joy", "sadness", "anger", "surprise", "fear"]
    label = torch.tensor([[label_list.index(label)]])  # Batch size 1

    # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to('cuda')
    label = label.to('cuda')

    #
    with torch.no_grad():
        outputs = model(tokens_tensor, labels=label)
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
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', config=pretrainedConfig)
# model.eval()  # Sets the module in evaluation mode(评估模式).
# To train the model, you should first set it back in training mode with model.train()
# If you have a GPU, put everything on cuda
model = model.to('cuda')

#
with open(r"./data/trainset.txt") as f:
    for line in f:
        l = line.rstrip().split('##')
        fine_tuning(l[0], l[1])
