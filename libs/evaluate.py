import torch

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            image, label = batch['images'], batch['labels']
            image = image.to(device, dtype=torch.float32, memory_format=torch.channels_last)
            label = label.to(device, dtype=torch.long)
            label_pred = model(image)
            loss = criterion(label_pred, label)
            losses.append(loss.item())
            _, predicted = torch.max(label_pred, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    avg_loss = sum(losses) / len(losses)
    accuracy = correct / total
    model.train()
    return avg_loss, accuracy
