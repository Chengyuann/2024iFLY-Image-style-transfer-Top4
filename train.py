# train.py

import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from model import get_model

def validation(model, loader, loss_fn):
    model.eval()
    losses = []
    with torch.no_grad():
        for image, target in loader:
            image, target = image.to('cuda'), target.to('cuda').float()
            output = model(image)
            loss = loss_fn(output, target)
            losses.append(loss.item())
    return np.mean(losses)

def train_model(train_loader, val_loader, model, epochs=40):
    model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        start_time = time.time()
        for image, target in tqdm(train_loader):
            image, target = image.to('cuda'), target.to('cuda').float()
            optimizer.zero_grad()
            output = model(image)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        train_losses.append(np.mean(losses))
        val_losses.append(validation(model, val_loader, loss_fn))
        scheduler.step()

        print(f"Epoch {epoch}/{epochs}")
        print(f"Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}")
        print(f"Time: {(time.time() - start_time) / 60:.2f} minutes\n")

    return train_losses, val_losses

def plot_losses(train_losses, val_losses, epochs):
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, 'b', label='Training loss')
    plt.plot(range(1, epochs + 1), val_losses, 'r', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_img = glob.glob('./train/*_input.jpg')
    train_mask = glob.glob('./train/*_target.jpg')
    train_img.sort()
    train_mask.sort()
    train_loader, val_loader = get_dataloaders(train_img, train_mask)
    model = get_model()
    train_losses, val_losses = train_model(train_loader, val_loader, model)
    plot_losses(train_losses, val_losses, 40)
