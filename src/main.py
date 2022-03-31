import torch
import utils
import time
import numpy as np
import os

from dataloader import TemperatureDataset, collate_fn, visualize_batch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from logger import Logger
from sklearn.metrics import accuracy_score
from arguments import parse_args
from PIL import Image

def evaluate(model, loss, val_loader, epoch, L):
    for img, target in val_loader:
        img = img.cuda()
        target = target.cuda()

        prediction = model(img)

        if isinstance(loss, torch.nn.CrossEntropyLoss):
            val_loss = loss(prediction, target)
            acc = accuracy_score(np.argmax(prediction.cpu().numpy(), axis=1), target.cpu().numpy())
        else:
            val_loss = loss(prediction, target.float())
            acc = accuracy_score(np.round(prediction.cpu().numpy()), target.cpu().numpy())

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

            if isinstance(loss, torch.nn.CrossEntropyLoss):
                train_loss = loss(prediction, target)
            else:
                train_loss = loss(prediction, target.float())

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

def test(model, loss, test_loader, work_dir):
    test_acc, test_loss = 0.0, 0.0
    save = True
    for i, (img, target) in enumerate(test_loader):
        img = img.cuda()
        target = target.cuda()
        prediction = model(img)

        if isinstance(loss, torch.nn.CrossEntropyLoss):
            _loss = loss(prediction, target)
            acc = accuracy_score(np.argmax(prediction.cpu().numpy(), axis=1), target.cpu().numpy())
            if save:
                correct_id = np.where(np.argmax(prediction.cpu().numpy(), axis=1) == target.cpu().numpy())[0]
                incorrect_id = np.where(np.argmax(prediction.cpu().numpy(), axis=1) != target.cpu().numpy())[0]

                try:
                    correct_img = img[correct_id[0]].cpu().numpy().astype(np.uint8).transpose(1, 2, 0)*255
                    im = Image.fromarray(correct_img)
                    fname = str(np.argmax(prediction.cpu().numpy(), axis=1)[correct_id[0]]) + \
                            "_" + str(target.cpu().numpy()[correct_id[0]])
                    im.save(os.path.join(work_dir, "correct_samples", fname+".png"))
                except:
                    pass

                try:
                    incorrect_img = img[incorrect_id[0]].cpu().numpy().astype(np.uint8).transpose(1, 2, 0)*255
                    im = Image.fromarray(incorrect_img)
                    fname = str(np.argmax(prediction.cpu().numpy(), axis=1)[incorrect_id[0]]) + \
                            "_" + str(target.cpu().numpy()[incorrect_id[0]])
                    im.save(os.path.join(work_dir, "incorrect_samples", fname + ".png"))
                except:
                    pass

        else:
            _loss = loss(prediction, target.float())
            acc = accuracy_score(np.round(prediction.cpu().numpy()), target.cpu().numpy())

        test_acc += (1 / (i + 1)) * (acc - test_acc)
        test_loss += (1 / (i + 1)) * (_loss - test_loss)

    return test_loss.item(), test_acc

def get_data(root):
    dset = TemperatureDataset(root, overfit=bool(args.overfit))
    test_dset = TemperatureDataset(root, train=False, overfit=args.overfit)

    v_split = 0.2

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

    if args.model_type == "small":
        from models.cnn_small import SmallCnn as MyModel
    else:
        from models.cnn import Cnn as MyModel

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    DATA_ROOT = args.data_root
    train_loader, val_loader, test_loader = get_data(DATA_ROOT)

    model = MyModel(args)
    model = model.cuda()

    if args.classification:
        loss = torch.nn.CrossEntropyLoss()
    else:
        loss = torch.nn.MSELoss()

    work_dir = os.path.join(args.log_dir, args.exp_name, args.seed)
    print("Working Directory ", work_dir)
    utils.make_dir(work_dir)
    utils.make_dir(os.path.join(work_dir, "correct_samples"))
    utils.make_dir(os.path.join(work_dir, "incorrect_samples"))
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
        test_loss, test_acc = test(model, loss, test_loader, work_dir)

    print("*"*25)
    print("Test Loss: ", test_loss)
    print("Test Acc: ", test_acc*100)
    print("*" * 25)

    f = open(os.path.join(work_dir, "test_results.txt"), 'w')
    f.write("Test Loss: " + str(test_loss))
    f.write("\nTest Acc: " + str(test_acc*100))
    f.close()
