import tensorflow as tf
import argparse
from PIL import Image
from os import listdir
from tqdm import tqdm
import numpy as np
from tensorflow.keras.utils import to_categorical
from random import shuffle

from model import create_model
from preprocess import check_paths, preprocessing
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--preprocess',
        type=bool,
        default=False,
        help='Whether data needs to be preprocessed'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='../data/',
        help='Relative path for directory containing images (e.g. this_path/train/real/00000.jpg)'
    )
    parser.add_argument(
        '--truncate_block_num',
        type=int,
        default=None,
        help='Block number to truncate after; omit to use full model'
    ),
    parser.add_argument(
        '--percent_of_data',
        type=float,
        default=100,
        help='amount of data to use in the model (e.g. 50 yields 25,000 train and 5,000 valid/test for each true/false)'
    )

    return parser.parse_args()

def load_imgs(path, percent_of_data):
    files = listdir(path)
    shuffle(files)
    files = files[:int(len(files)*percent_of_data/100)]
    imgs = []
    for fp in tqdm(files, total=len(files)):
        # changed this part for concurrency memory issue
        temp = Image.open(f'{path}/{fp}')
        keep = temp.copy()
        imgs.append(np.array(keep))
        temp.close()
    return np.array(imgs)

def load_data(data_type, data_path, percent_of_data):
    # added if condition because so far we're only preprocessing train
    train_path = f'{data_path}/{data_type}/preprocess/'
    (reals, fakes) = [load_imgs(train_path + t, percent_of_data) for t in ['real', 'fake']]
    return {
        'data': np.vstack([reals, fakes]),
        'labels': to_categorical(np.hstack([np.ones((reals.shape[0])), np.zeros((fakes.shape[0]))]))
    }

def main():
    args = parse_args()
    if args.preprocess:
        check_paths(args.data_path)
        preprocessing(args.data_path)
    data = {t: load_data(t, args.data_path, args.percent_of_data) for t in ['train', 'valid', 'test']}
    model = create_model(args.truncate_block_num)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.045, momentum=0.9)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(data['train']['data'], data['train']['labels'], batch_size=1000)  # also batch_size and epochs
    print(model.evaluate(data['test']['data'], data['test']['labels']))


if __name__ == "__main__":
    main()
