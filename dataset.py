import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import prepare_data
from albumentations.torch.functional import img_to_tensor


class RoboticsDataset(Dataset):
    def __init__(self, file_names, to_augment=False, transform=None, mode='train'):
        self.file_names = file_names
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)
        mask = load_mask(img_file_name)

        data = {"image": image, "mask": mask}
        augmented = self.transform(**data)
        image, mask = augmented["image"], augmented["mask"]

        assert np.array([el in [0, 1] for el in np.unique(mask)]).all()
        # print(np.unique(mask))

        if self.mode == 'train':
            return img_to_tensor(image), torch.from_numpy(np.expand_dims(mask, 0)).float()
        else:
            return img_to_tensor(image), str(img_file_name)


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path):
    mask = cv2.imread((str(path).replace('train_v2', 'binary_masks')), 0)
    return (mask / prepare_data.binary_factor).astype(np.uint8)
