import json
import torch
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, loss_function, len_test, train_loss, train_acc, epoch):
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for input_ids, attention_mask, labels, _ in test_loader:
            outputs = model(input_ids, attention_mask)
            loss = loss_function(outputs, labels)
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            total_acc += (predictions == labels).sum().item()

    test_loss = total_loss/len(test_loader)
    test_acc = total_acc/len_test*100

    print(f"Epoch {epoch} -> Train loss: {train_loss:.4f} Train acc: {train_acc:.2f}% Test acc: {test_acc:.2f}%")
    
    return test_loss, test_acc

def train_model(model, optimizer, loss_function, train_loader, len_train, test_loader, len_test, epoch):
    model.train()
    total_loss = 0
    total_acc = 0

    for input_ids, attention_mask, labels, _ in train_loader:
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

    test_loss, test_acc = evaluate_model(model, test_loader, loss_function, len_test, train_loss, train_acc, epoch)
    # print(f"Training loss: {total_loss/len(train_loader)}")

    return train_loss, train_acc, test_loss, test_acc

def plot_loss(train, title, train_label="", y_label="", save_path=None):
    epochs = range(1, len(train) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train, label=train_label)
    
    plt.xlabel("Epochs")
    plt.ylabel(y_label)
    plt.title(title)
    # plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()

def plot_accuracies(train, test, title, train_label="", test_label="", y_label="", save_path=None):
    epochs = range(1, len(train) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train, label=train_label)
    plt.plot(epochs, test, label=test_label)
    
    plt.xlabel("Epochs")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()

def get_failed_examples(model, test_loader, tokenizer):
    model.eval()
    total_acc = 0
    failed_examples = []

    with torch.no_grad():
        for input_ids, attention_mask, labels, article_links in test_loader:
            outputs = model(input_ids, attention_mask)
            predictions = torch.argmax(outputs, dim=1)
            total_acc += (predictions == labels).sum().item()

            for i in range(len(labels)):
                if predictions[i] != labels[i]:
                    input_id_list = input_ids[i].tolist()
                    input_id_list = [token_id for token_id in input_id_list if token_id != 0]
                    decoded_sentence = tokenizer.decode(input_id_list, skip_special_tokens=True)
                    failed_examples.append({
                        "input_ids": input_id_list,
                        "label": labels[i].item(),
                        "prediction": predictions[i].item(),
                        "decoded_sentence": decoded_sentence,
                        "article_link": article_links[i]
                    })
    
    return failed_examples

def save_failed_examples(failed_examples, file_path):
    with open(file_path, "w") as f:
        json.dump(failed_examples, f, indent=4)