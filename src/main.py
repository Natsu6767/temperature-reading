import torch
import utils
import time
import numpy as np
import os

from dataloader import TemperatureDataset, collate_fn, visualize_batch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from models.cnn_small import MyModelSmall as MyModel
from logger import Logger
from sklearn.metrics import accuracy_score
from arguments import parse_args
from models.resnet import ResNet18

def evaluate(model, loss, val_loader, epoch, L):
    for img, target in val_loader:
        img = img.cuda()
        target = target.cuda()

        prediction = model(img)

        acc = accuracy_score(np.argmax(prediction.cpu().numpy(), axis=1), target.cpu().numpy())
        val_loss = loss(prediction, target)

        L.log("eval/loss", val_loss, epoch)
        L.log("eval/acc", acc, epoch)

def train(model, loss, train_loader, val_loader, L, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    start_time = time.time()

    for epoch in range(args.epochs):
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
            evaluate(model, loss, val_loader, epoch, L)
            L.dump(epoch)
            model.train()

def test(model, loss, test_loader):
    test_acc, test_loss = 0.0, 0.0
    for i, (img, target) in enumerate(test_loader):
        img = img.cuda()
        target = target.cuda()
        prediction = model(img)

        acc = accuracy_score(np.argmax(prediction.cpu().numpy(), axis=1), target.cpu().numpy())
        _loss = loss(prediction, target)

        test_acc += (1 / (i + 1)) * (acc - test_acc)
        test_loss += (1 / (i + 1)) * (_loss - test_loss)

    return test_loss.item(), test_acc

def get_data(root):
    dset = TemperatureDataset(root, overfit=bool(args.overfit))
    test_dset = TemperatureDataset(root, train=False, overfit=args.overfit)

    #if not args.overfit:
    v_split = 0.2
    #else:
    #    v_split = 0.0

    dataset_size = len(dset)
    indices = list(range(dataset_size))
    split = int(np.floor(v_split * dataset_size))

    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dset, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(dset, batch_size=args.batch_size, sampler=val_sampler)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=True)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    DATA_ROOT = args.data_root
    train_loader, val_loader, test_loader = get_data(DATA_ROOT)

    #model = MyModel()
    model = ResNet18()
    model = model.cuda()
    loss = torch.nn.CrossEntropyLoss()

    work_dir = os.path.join("logs", "debug")
    print("Working Directory ", work_dir)
    utils.make_dir(work_dir)
    utils.write_info(args, os.path.join(work_dir, "info.log"))
    model_dir = utils.make_dir(os.path.join(work_dir, "model"))
    L = Logger(work_dir)

    train(model, loss, train_loader, val_loader, L, args)

    print("MODEL FINISHED TRAINING!")
    print("Saving Model!")
    torch.save(
        model.state_dict(), os.path.join(model_dir, "final_model.pt"))
    print("\n")
    print("Testing Model!!")

    model.eval()
    with torch.no_grad():
        test_loss, test_acc = test(model, loss, test_loader)

    print("*"*25)
    print("Test Loss: ", test_loss)
    print("Test Acc: ", test_acc)
    print("*" * 25)
