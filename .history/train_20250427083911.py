import torch
from config import learning_rate, epochs
from utils import set_lr, clip_gradient, progress_bar

def train_model(model, num_epochs, train_loader, loss_fn, optimizer, device):
    model.to(device)
    for epoch in range(num_epochs):
        print(f"\nEpoch: {epoch}")
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            clip_gradient(optimizer, 0.1)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

            progress_bar(batch_idx, len(train_loader), f"Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total})")