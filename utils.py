import torch.nn as nn
from tqdm import tqdm
import torch

def overfit_single_batch(model, loss_fn, optimizer, train_loader, epochs=20):
    model.train()
    x, y = next(iter(train_loader))
    for epoch in range(epochs):
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean().item() * 100
        print(f"Epoch {epoch+1:02d} | Loss: {loss.item():.4f} | Accuracy: {acc:.2f}%")

    for name, param in model.named_parameters():
        if param.grad is not None:
            print(name, param.grad.norm().item())

def alexnet_gaussian_init(m):
    """We initialized the weights in each layer from a zero-mean Gaussian distribution with standard deviation 0.01."""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)



def train(model: nn.Module, optimizer, loss_fn, epochs, train_loader, device):

    for epoch in range(epochs):
        local_loss = 0
        accumulation = 0
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accumulation += 1
            local_loss += loss.item()
        print(f"Epoch {epoch + 1}, Train Loss: {local_loss/accumulation:.4f}")
    return model

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            logits = model(x)
            _, pred = torch.max(logits, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    model.train()
    print(f"Test Accuracy: {100 * correct // total}%")