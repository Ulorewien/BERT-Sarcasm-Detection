import torch

def evaluate_model(model, test_loader, loss_function, len_test, train_loss, train_acc):
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

    test_loss = total_loss/len(test_loader)
    test_acc = total_acc/len_test*100

    print(f"Train loss: {train_loss:.4f} Train acc: {train_acc:.2f}% Test loss: {test_loss:.4f} Test acc: {test_acc:.2f}%")
    
    return test_loss, test_acc

def train_model(model, optimizer, loss_function, train_loader, len_train, test_loader, len_test):
    model.train()
    total_loss = 0
    total_acc = 0

    for input_ids, attention_mask, labels in train_loader:
        outputs = model(input_ids, attention_mask)
        loss = loss_function(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        total_acc += (predictions == labels).sum().item()

    train_loss = total_loss/len(train_loader)
    train_acc = total_acc/len_train*100

    test_loss, test_acc = evaluate_model(model, test_loader, loss_function, len_test, train_loss, train_acc)
    # print(f"Training loss: {total_loss/len(train_loader)}")

    return train_loss, train_acc, test_loss, test_acc