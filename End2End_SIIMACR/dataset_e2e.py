import os
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image


class DatasetGaze_e2e(Dataset):
    def __init__(self, data_dir, csv_path, phase, img_size):
        img_folder = data_dir + '/img/'
        gaze_folder = data_dir + '/gaze/'
        img_paths = [img_folder + path for path in os.listdir(img_folder)]
        gaze_paths = [gaze_folder + path for path in os.listdir(gaze_folder)]
        img_paths.sort()
        gaze_paths.sort()

        self.csv = pd.read_csv(csv_path, usecols=[0, 2])
        self.img_paths = img_paths
        self.gaze_paths = gaze_paths
        self.phase = phase
        self.transform = get_transform(phase, img_size)

    def __getitem__(self, index):
        img_path, gaze_path = self.img_paths[index], self.gaze_paths[index]
        img = np.array(Image.open(img_path))
        gaze = np.array(Image.open(gaze_path))

        if self.transform is not None:
            transformed = self.transform(image=img, mask=gaze)
            img = transformed["image"] / 255.
            gaze = transformed["mask"] / 255.
            gaze = gaze.unsqueeze(0)

        name = img_path.split('/')[-1].split('.png')[0]
        class_id = self.csv.loc[self.csv.image_id == name].class_id
        class_id = int(class_id.iloc[0])
        if class_id == 14:
            cls = 0
        else:
            cls = 1
        return img, gaze, cls, img_path

    def __len__(self):
        return len(self.img_paths)


def get_transform(phase, img_size):
    if phase == 'train':
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.HorizontalFlip(),
                A.ShiftScaleRotate(rotate_limit=15),
                A.RandomBrightnessContrast(),
                A.RandomGamma(),
                A.Normalize(mean=0.456, std=0.224),
                ToTensorV2(),
            ])
    else:
        return A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(mean=0.456, std=0.224),
                ToTensorV2(),
            ])