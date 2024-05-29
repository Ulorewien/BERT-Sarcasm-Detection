import numpy as np
import pandas as pd
import transformers
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import NewsHeadlinesDataset
from model import SarcasmDetectionModel
from util import train_model, plot_progress

# Global Variables
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
news_dataset_dir = "Datasets\\News_Headlines\\Sarcasm_Headlines_Dataset_v2.json"
split_ratio = 0.8
model_save_path = "first_run.pth"
loss_save_path = "losses.png"
acc_save_path = "accuracies.png"

# Hyperparameters
len_dataset = 2000
batch_size = 32
lr = 2e-5
n_epoch = 20

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
attention_mask_news = np.where(padded_dataset_news != 0, 1, 0)
print(f"Shape of the masked dataset: {attention_mask_news.shape}")

# Split the data in training and testing sets and create dataloaders
split_val = int(split_ratio*len_dataset)
print(f"Splitting the data -> Train ({split_val}) & Test ({len_dataset-split_val})")

train_features_news = torch.Tensor(padded_dataset_news[:split_val], device=device)
train_mask_news = torch.Tensor(attention_mask_news[:split_val], device=device)
train_labels_news = torch.Tensor(news_dataset["is_sarcastic"].values[:split_val], device=device)
train_dataset_news = NewsHeadlinesDataset(train_features_news, train_mask_news, train_labels_news)
train_loader_news = DataLoader(train_dataset_news, batch_size=batch_size, shuffle=True)

test_features_news = torch.Tensor(padded_dataset_news[split_val:], device=device)
test_mask_news = torch.Tensor(attention_mask_news[split_val:], device=device)
test_labels_news = torch.Tensor(news_dataset["is_sarcastic"].values[split_val:], device=device)
test_dataset_news = NewsHeadlinesDataset(test_features_news, test_mask_news, test_labels_news)
test_loader_news = DataLoader(test_dataset_news, batch_size=batch_size)

# Define the model, optimizer and the loss function
model = SarcasmDetectionModel(bert_model).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss()

# Create arrays to document progress
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# Train and Evaluate the model
for epoch in range(n_epoch):
    train_loss, train_acc, test_loss, test_acc = train_model(model, optimizer, loss_function, train_loader_news, len(train_labels_news), test_loader_news, len(test_labels_news))
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

# Save the model
torch.save(model.state_dict(), model_save_path)

# Plot the results
plot_progress(train_losses, test_losses, "Loss Plot", "Train Loss", "Test Loss", "Loss", loss_save_path)
plot_progress(train_accuracies, test_accuracies, "Accuracy Plot", "Train Acc", "Test Acc", "Accuracy", acc_save_path)
