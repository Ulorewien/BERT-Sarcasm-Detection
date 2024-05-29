import numpy as np
import pandas as pd
import transformers
from torch.utils.data import DataLoader
from datasets import NewsHeadlinesDataset
from model import SarcasmDetectionModel

# Global Variables
news_dataset_dir = "Datasets\\News_Headlines\\Sarcasm_Headlines_Dataset_v2.json"
split_ratio = 0.8

# Hyperparameters
len_dataset = 2000
batch_size = 32

# Load the data
print("Loading data...")
news_dataset = pd.read_json(news_dataset_dir, lines=True)
news_dataset = news_dataset[:len_dataset]
print(f"Instances of each sample in the entire dataset: {news_dataset["is_sarcastic"].value_counts()}")

# Load the pre-trained BERT model
model_class, tokenizer_class, pretrained_weights = (transformers.BertModel, transformers.BertTokenizer, "bert-base-uncased")

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
bert_model = model_class.from_pretrained(pretrained_weights)

# Tokenize the dataset
tokenized_dataset_news = news_dataset["headline"].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

# Padding to match the shapes of all the input data
max_len = 0
for i in tokenized_dataset_news.values:
    if len(i) > max_len:
        max_len = len(i)

padded_dataset_news = np.array([i + [0]*(max_len-len(i)) for i in tokenized_dataset_news.values])
print(f"Shape of the padded dataset: {np.array(padded_dataset_news).shape}")

# Add a mask so that the model doesn't consider padded tokens
masked_dataset_news = np.where(padded_dataset_news != 0, 1, 0)
print(f"Shape of the masked dataset: {masked_dataset_news.shape}")

# Split the data in training and testing sets and create dataloaders
split_val = int(split_ratio*len_dataset)
print(f"Splitting the data -> Train ({split_val}) & Test ({len_dataset-split_val})")
train_features_news = masked_dataset_news[:split_val]
test_features_news = masked_dataset_news[split_val:]
train_labels_news = news_dataset["is_sarcastic"].values[:split_val]
test_labels_news = news_dataset["is_sarcastic"].values[split_val:]

train_dataset_news = NewsHeadlinesDataset(train_features_news, train_labels_news)
test_dataset_news = NewsHeadlinesDataset(test_features_news, test_labels_news)
train_loader_news = DataLoader(train_dataset_news, batch_size=batch_size, shuffle=True)
test_loader_news = DataLoader(test_dataset_news, batch_size=batch_size)