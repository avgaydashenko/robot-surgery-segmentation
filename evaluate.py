from pathlib import Path
import argparse
import cv2
import numpy as np


def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--train_path', type=str, default='/home/raznem/proj_kaggle_airbus/data',
        help='path where train images with ground truth are located')
    arg('--target_path', type=str, default='predictions/unet11', help='path with predictions')
    args = parser.parse_args()

    result_dice = []
    result_jaccard = []

    for file_name in (Path(args.train_path) / 'binary_masks').glob('*'):
        y_true = (cv2.imread(str(file_name), 0) > 0).astype(np.uint8)

        pred_file_name = (Path(args.target_path) / 'binary' / file_name.name)

        pred_image = (cv2.imread(str(pred_file_name), 0) > 255 * 0.5).astype(np.uint8)
        y_pred = pred_image

        result_dice += [dice(y_true, y_pred)]
        result_jaccard += [jaccard(y_true, y_pred)]

    print('Dice = ', np.mean(result_dice), np.std(result_dice))
    print('Jaccard = ', np.mean(result_jaccard), np.std(result_jaccard))
