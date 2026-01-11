import math
from tqdm import tqdm

import torch

from libs.evaluate import evaluate_model

def lr_scheduler(optimizer, epoch, n_epochs):
    if epoch < 12:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-4
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * math.cos(epoch / n_epochs)

    return optimizer

def train_one_epoch(model, train_dataloader, optimizer, criterion, device, epoch, n_epochs, n_train, train_losses):
    model.train()
    epoch_loss = 0.0
    with tqdm(total=n_train, desc=f'Epoch {epoch}/{n_epochs}', unit='img') as pbar:
        for batch in train_dataloader:
            image, label = batch['images'], batch['labels']
            image = image.to(device, dtype=torch.float32, memory_format=torch.channels_last)
            label = label.to(device, dtype=torch.long)

            label_pred = model(image)
            loss = criterion(label_pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.update(image.shape[0])
            epoch_loss += loss.item()
            pbar.set_postfix(**{f'loss (batch)': loss.item()})

        train_losses.append(epoch_loss / len(train_dataloader))

def train_model(model, train_dataloader, val_dataloader, optimizer, criterion, config, n_epochs, n_train):
    model.to(config['DEVICE'])
    train_losses = []
    val_losses = []
    val_accuracy = []

    best_model = model
    best_val_acc = 0.0
    best_val_loss = 0.0
    for epoch in range(1, n_epochs+1):
        train_one_epoch(model, train_dataloader, optimizer, criterion, config['DEVICE'], epoch, n_epochs, n_train, train_losses)
        val_loss, val_acc = evaluate_model(model, val_dataloader, criterion, config['DEVICE'])
        optimizer = lr_scheduler(optimizer, epoch, n_epochs)
        val_losses.append(val_loss)
        val_accuracy.append(val_acc)
        print(f'Validation loss: {val_loss}, Validation accuracy: {val_acc}')
        if val_acc > best_val_acc:
            best_model = model
            best_val_acc = val_acc
            best_val_loss = val_loss
        elif val_acc == best_val_acc and best_val_loss > val_loss:
            best_model = model
            best_val_acc = val_acc
            best_val_loss = val_loss

    return best_model, train_losses, val_losses, val_accuracy

