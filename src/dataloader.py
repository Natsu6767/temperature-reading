import os
import torch
import numpy as np
import glob
import json
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision import transforms, utils
from PIL import Image


def collate_fn(batch):
    imgs = [data[0] for data in batch]
    target = [data[1] for data in batch]
    target = torch.LongTensor(target)

    return [imgs, target]

def visualize_batch(imgs):
    grid_imgs = utils.make_grid(imgs, nrow=8)
    plt.imshow(grid_imgs.permute(1, 2, 0))
    plt.savefig("batch.png")
class TemperatureDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, overfit=False):
        print("In Dataset")
        self.root_dir = root_dir
        train_test = "train" if train else "test"
        image_path = os.path.join(root_dir, train_test, "images")
        gt_path = os.path.join(root_dir, train_test, "targets")

        #import pdb; pdb.set_trace()
        if transform is not None:
            self.transforms = transform
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                #transforms.RandomCrop((500, 84)),
                transforms.Resize((256, 256))])

        self.images = []
        self.targets = []

        i = 0
        for file in glob.glob(os.path.join(image_path,"*.png")):
            if overfit and i > 1000:
                break
            open_image = Image.open(file)
            open_image = np.array(open_image)
            self.images.append(open_image)

            target_path = os.path.join(gt_path, file.split("/")[-1]+".json")
            f = open(target_path)
            data = json.load(f)
            self.targets.append(data["relative_reading"])
            f.close()

            i += 1

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.images[idx]
        target = self.targets[idx]

        tensor_image = self.transforms(img)

        return tensor_image, target
