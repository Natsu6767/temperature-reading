import torch
import utils
import time
import numpy as np
import os

from dataloader import TemperatureDataset, collate_fn, visualize_batch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from models.cnn import MyModel
from logger import Logger
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    DATA_ROOT = "../thermometers"
    print(DATA_ROOT)
    dset = TemperatureDataset(DATA_ROOT, overfit=False)

    if True:
        v_split = 0.2
    else:
        v_split = 0.0
    dataset_size = len(dset)
    indices = list(range(dataset_size))
    split = int(np.floor(v_split * dataset_size))

    np.random.shuffle(indices)

    train_indices, val_indices =  indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dset, batch_size=32, sampler=train_sampler)
    val_loader = DataLoader(dset, batch_size=32, sampler=val_sampler)

    model = MyModel()
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #loss = torch.nn.MSELoss()
    loss = torch.nn.CrossEntropyLoss()
    print("Looping")

    work_dir = os.path.join("logs", "debug")
    print("Working Directory ", work_dir)
    utils.make_dir(work_dir)
    model_dir = utils.make_dir(os.path.join(work_dir, "model"))

    L = Logger(work_dir)

    model.train()
    start_time = time.time()
    for epoch in range(10):
        iters = 0
        for img, target in train_loader:
            iters += 1
            img = img.cuda()
            target = target.cuda()

            optimizer.zero_grad()
            import pdb;

            #pdb.set_trace()
            prediction = model(img)
            train_loss = loss(prediction, target)


            L.log("train/loss", train_loss, epoch)

            train_loss.backward()
            optimizer.step()

        L.log("train/duration", time.time() - start_time, epoch)
        start_time = time.time()
        L.dump(epoch)
        with torch.no_grad():
            model.eval()
            for img, target in val_loader:
                img = img.cuda()
                target = target.cuda()
                prediction = model(img)

                acc = accuracy_score(np.argmax(prediction.cpu().numpy(), axis=1), target.cpu().numpy())

                val_loss = loss(prediction, target)
                L.log("eval/loss", val_loss, epoch)
                L.log("eval/acc", acc, epoch)


            L.dump(epoch)

            model.train()
