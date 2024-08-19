# dataset.py

import cv2
import glob
import numpy as np
import torch.utils.data as D
import torchvision.transforms as T
import albumentations as A

class FoodDataset(D.Dataset):
    def __init__(self, images, masks, transform):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.as_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize([0.625, 0.448, 0.688], [0.131, 0.177, 0.101]),
        ])

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        mask = cv2.imread(self.masks[index])
        mask = np.transpose(mask, (2, 0, 1))
        return self.as_tensor(image), mask / 255.0

    def __len__(self):
        return len(self.images)

def get_dataloaders(train_img, train_mask, batch_size=4):
    train_ds = FoodDataset(train_img[:-200], train_mask[:-200], transform=trfm)
    val_ds = FoodDataset(train_img[-200:], train_mask[-200:], transform=trfm)
    train_loader = D.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=3)
    val_loader = D.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=3)
    return train_loader, val_loader
