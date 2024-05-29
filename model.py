from torch import nn

class SarcasmDetectionModel(nn.Module):
    def __init__(self, bert_model, n_classes=2):
        super().__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(768, n_classes)

    def forward(self, x):
        x = self.bert(x)
        x = self.classifier(x)
        return x