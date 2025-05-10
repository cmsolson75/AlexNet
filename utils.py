import torch.nn as nn

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