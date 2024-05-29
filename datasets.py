from torch.utils.data import Dataset

class NewsHeadlinesDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.headlines = df["headline"].values
        self.labels = df["is_sarcastic"].values

    def __len__(self):
        return len(self.headlines)
    
    def __getitem__(self, index):
        return self.headlines[index], self.labels[index]
