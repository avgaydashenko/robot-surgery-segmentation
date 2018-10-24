"""
[1] Merge masks with different instruments into one binary mask
[2] Crop black borders from images and masks
"""
from pathlib import Path

from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd


data_path = Path('/home/raznem/proj_kaggle_airbus/data')
train_path = data_path / 'train_v2'
binary_factor = 255


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


if __name__ == '__main__':

    masks = pd.read_csv('/home/raznem/proj_kaggle_airbus/data/train_ship_segmentations_v2.csv')

    binary_mask_folder = (train_path / 'binary_masks')
    binary_mask_folder.mkdir(exist_ok=True, parents=True)

    for file_name in tqdm(list(train_path.glob('*'))):
        if 'jpg' not in str(file_name):
            continue

        ImageId = str(file_name)[-13:]
        img_masks = masks.loc[masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()

        all_masks = np.zeros((768, 768))

        if img_masks != [np.nan]:
            for mask in img_masks:
                all_masks += rle_decode(mask)

        mask_binary = all_masks.astype(np.uint8) * binary_factor

        cv2.imwrite(str(binary_mask_folder / file_name.name), mask_binary)
