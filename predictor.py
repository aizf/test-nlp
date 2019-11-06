from fast_bert.prediction import BertClassificationPredictor
DATA_PATH = r'./data'
LABEL_PATH = DATA_PATH
OUTPUT_DIR = r"./output_dir"
MODEL_PATH = OUTPUT_DIR + r'/model_out'

predictor = BertClassificationPredictor(
    model_path=MODEL_PATH,
    label_path=LABEL_PATH,  # location for labels.csv file
    multi_label=True,
    model_type='bert',
    do_lower_case=True)

# Batch predictions
texts = [
    'I really love the Netflix original movies',
    'this movie is not worth watching',
    'I am going to have to become someone else', r"You can't change the past.",
    r"We got married before you did, and we still don't have children.",
    "i am so happy", "You just stay away from me please.",
    "Everything you ever told me was a lie.",
    "We're the luckiest sons-of-bitches in the world."
]

multiple_predictions = predictor.predict_batch(texts)
for i in multiple_predictions:
    print(i)
