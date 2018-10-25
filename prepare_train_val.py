from random import shuffle
from prepare_data import train_path


def get_split():

    all_files = list(train_path.glob('*'))
    shuffle(all_files)

    num = int(len(all_files) / 4)

    val_file_names = all_files[:num]
    train_file_names = all_files[num:]

    return train_file_names, val_file_names
