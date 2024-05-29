import torch

def train_model(model, optimizer, train_loader, loss_function):
    model.train()
    total_loss = 0

    for input_ids, attention_mask, labels in train_loader:
        outputs = model(input_ids, attention_mask)
        loss = loss_function(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Training loss: {total_loss/len(train_loader)}")

def evaluate_model(model, test_loader, loss_function):
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            outputs = model(input_ids, attention_mask)
            loss = loss_function(outputs, labels)
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            total_acc += (predictions == labels).sum().item()

    print(f"Test loss: {total_loss/len(test_loader)} Test acc: {total_acc/len(test_loader)*100}%")