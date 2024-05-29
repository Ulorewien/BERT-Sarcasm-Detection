from torch import nn

class SarcasmDetectionModel(nn.Module):
    def __init__(self, bert_model, n_classes=2):
        super().__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(768, n_classes)

    def forward(self, x, attention_mask):
        x = self.bert(x, attention_mask=attention_mask)
        x = x[0][:,0,:]
        x = self.classifier(x)
        return x