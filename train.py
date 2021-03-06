import argparse
import json
from pathlib import Path
from validation import validation_binary

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.backends.cudnn

from models import UNet11, LinkNet34, UNet, UNet16, AlbuNet
from loss import LossBinary
from dataset import RoboticsDataset
import utils
from prepare_train_val import get_split

from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    RandomCrop,
    CenterCrop,
    Transpose,
    RandomRotate90,
    RandomSizedCrop
)

moddel_list = {'UNet11': UNet11,
               'UNet16': UNet16,
               'UNet': UNet,
               'AlbuNet': AlbuNet,
               'LinkNet34': LinkNet34}


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--jaccard-weight', default=0.5, type=float)
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')
    arg('--root', default='runs/debug', help='checkpoint root')
    arg('--batch-size', type=int, default=16)
    arg('--n-epochs', type=int, default=10)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=12)
    arg('--train_crop_height', type=int, default=256)
    arg('--train_crop_width', type=int, default=256)
    arg('--val_crop_height', type=int, default=768)
    arg('--val_crop_width', type=int, default=768)
    arg('--model', type=str, default='AlbuNet', choices=moddel_list.keys())

    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    num_classes = 1

    if args.model == 'UNet':
        model = UNet(num_classes=num_classes)
    else:
        model_name = moddel_list[args.model]
        model = model_name(num_classes=num_classes, pretrained=True)

    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        raise SystemError('GPU device not found')

    loss = LossBinary(jaccard_weight=args.jaccard_weight)

    cudnn.benchmark = True

    def make_loader(file_names, shuffle=False, transform=None, batch_size=1):
        return DataLoader(
            dataset=RoboticsDataset(file_names, transform=transform),
            shuffle=shuffle,
            num_workers=args.workers,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available()
        )

    train_file_names, val_file_names = get_split()

    print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names)))

    def train_transform(p=1):
        return Compose([
            # CenterCrop(height=256, width=256, p=1),
            RandomCrop(height=args.train_crop_height, width=args.train_crop_width, p=1),
            Transpose(p=0.5),
            RandomRotate90(p=0.5),
            # RandomSizedCrop(min_max_height=(int(0.8 * args.train_crop_height), args.train_crop_height),
            #                 height=args.val_crop_height, width=args.val_crop_width),
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            Normalize(p=1)
        ], p=p)

    def val_transform(p=1):
        return Compose([
            CenterCrop(height=args.val_crop_height, width=args.val_crop_width, p=1),
            Normalize(p=1)
        ], p=p)

    train_loader = make_loader(train_file_names, shuffle=True, transform=train_transform(p=1),
                               batch_size=args.batch_size)
    valid_loader = make_loader(val_file_names, transform=val_transform(p=1), batch_size=len(device_ids))

    root.joinpath('params.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))

    utils.train(
        init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
        args=args,
        model=model,
        criterion=loss,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=validation_binary
    )


if __name__ == '__main__':
    main()
