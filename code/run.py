import tensorflow as tf
import argparse
from PIL import Image
from os import listdir
from tqdm import tqdm
import numpy as np

from model import create_model
from preprocess import check_paths, preprocessing

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
    )

    return parser.parse_args()

def load_imgs(path):
    files = listdir(path)
    imgs = []
    for fp in tqdm(files, total=len(files)):
        imgs.append(Image.open(f'{path}/{fp}'))
    return np.array(imgs)

def load_data(data_type, data_path):
    train_path = f'{data_path}/{data_type}/preprocess/'
    (reals, fakes) = [load_imgs(train_path + t) for t in ['real', 'fake']]
    return {
        'data': np.stack([reals, fakes], axis=0),
        'labels': np.stack([np.ones((reals.shape[0])), np.zeros((fakes.shape[0]))])
    }

def main():
    args = parse_args()
    if args.preprocess:
        check_paths(args.data_path)
        preprocessing(args.data_path)
    data = {t: load_data(t, args.data_path) for t in ['train', 'valid', 'test']}
    model = create_model(args.truncate_block_num)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.045, momentum=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(data['train']['data'], data['train']['labels'])  # also batch_size and epochs
    print(model.evaluate(data['test']['data'], data['test']['labels']))

if __name__ == '__main__':
    main()