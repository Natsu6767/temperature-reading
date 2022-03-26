import os
import torch
import numpy as np
import glob
import json

from torch.utils.data import Dataset
from torchvision import transforms, utils
from PIL import Image


def collate_fn(batch):
    imgs = [data[0] for data in batch]
    target = [data[1] for data in batch]
    target = torch.LongTensor(target)

    return [imgs, target]

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
            self.transforms = transforms.ToTensor()

        self.images = []
        self.targets = []

        i = 0
        for file in glob.glob(os.path.join(image_path,"*.png")):
            if overfit and i > 5:
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
