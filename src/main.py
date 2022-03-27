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
from arguments import parse_args

def evaluate(model, val_loader, L):
    for img, target in val_loader:
        img = img.cuda()
        target = target.cuda()

        prediction = model(img)

        acc = accuracy_score(np.argmax(prediction.cpu().numpy(), axis=1), target.cpu().numpy())
        val_loss = loss(prediction, target)

        L.log("eval/loss", val_loss, epoch)
        L.log("eval/acc", acc, epoch)

def train(model, train_loader, val_loader, L, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss = torch.nn.CrossEntropyLoss()
    model.train()
    start_time = time.time()

    for epoch in range(10):
        for img, target in train_loader:
            img = img.cuda()
            target = target.cuda()

            prediction = model(img)
            train_loss = loss(prediction, target)

            L.log("train/loss", train_loss, epoch)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        L.log("train/duration", time.time() - start_time, epoch)
        start_time = time.time()
        L.dump(epoch)

        with torch.no_grad():
            model.eval()
            evaluate(model, val_loader, L)
            L.dump(epoch)
            model.train()

def test(model, test_loader):
    test_acc, test_loss = 0.0, 0.0
    for i, (img, target) in enumerate(test_loader):
        img = img.cuda()
        target = target.cuda()
        prediction = model(img)

        acc = accuracy_score(np.argmax(prediction.cpu().numpy(), axis=1), target.cpu().numpy())
        _loss = loss(prediction, target)

        test_acc += (1 / (i + 1)) * (test_acc - acc)
        test_loss += (1 / (i + 1)) * (test_loss - _loss)

    return test_loss, test_acc

if __name__ == "__main__":
    args = parse_args()

    DATA_ROOT = args.data_root
    print(DATA_ROOT)

    dset = TemperatureDataset(DATA_ROOT, overfit=bool(args.overfit))
    test_dset = TemperatureDataset(DATA_ROOT, train=False)

    if args.overfit:
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

    train_loader = DataLoader(dset, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(dset, batch_size=args.batch_size, sampler=val_sampler)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=True)

    model = MyModel()
    model = model.cuda()

    work_dir = os.path.join("logs", "debug")
    print("Working Directory ", work_dir)
    utils.make_dir(work_dir)
    model_dir = utils.make_dir(os.path.join(work_dir, "model"))
    L = Logger(work_dir)

    print("MODEL FINISHED TRAINING!")
    print("Saving Model!")
    torch.save(
        model.state_dict(), os.path.join(model_dir, "final_model.pt"))
    print("\n")
    print("Testing Model!!")

    model.eval()
    with torch.no_grad():
        test_acc, test_loss = test(model, test_loader)

    print("*"*25)
    print("Test Loss: ", test_loss)
    print("Test Acc: ", test_acc)
    print("*" * 25)
