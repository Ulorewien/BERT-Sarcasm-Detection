from torch.utils.data import Dataset

class NewsHeadlinesDataset(Dataset):
    def __init__(self, features, attention_mask, labels):
        super().__init__()
        self.features = features
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index], self.attention_mask[index], self.labels[index]
