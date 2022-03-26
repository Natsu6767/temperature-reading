import torch

from dataloader import TemperatureDataset, collate_fn
from torch.utils.data import DataLoader

if __name__ == "__main__":
    DATA_ROOT = "../thermometers"
    print(DATA_ROOT)
    dset = TemperatureDataset(DATA_ROOT, overfit=True)
    train_loader = DataLoader(dset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    print("Looping")
    for img, target in train_loader:
        print(img[0].dtype, "\t", target, "\t", target.dtype)
