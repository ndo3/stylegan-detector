from os import listdir
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from PIL import Image
import argparse
from tqdm import tqdm
from random import shuffle

from model import create_model
from preprocess import check_paths


def parse_args():
    """Handles argument parsing for this file on the command line, including:
     - Path for data
     - The truncation location, if any, for the model (Chai et al. tried 
     truncating after block 1, 2, 3, 4, or 5)
     - The percent of data to use for each of train/validate/test
     - The size to resize each image to (replacing the preprocessing 
     function in preprocess.py)
     - The batch size
     - The number of epochs
     - The learning rate

    Returns:
        namespace: The values of the command line arguments.
    """
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        '--percent_of_data',
        type=float,
        default=100,
        help='Amount of data to use in the model (e.g. 50 yields 25,000 train and 5,000 valid/test for each true/false)'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=299,
        help='Width and height dimension of each photo to resize to before using the data'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=25,
        help='Number of images per batch'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=4,
        help='Number of epochs to train for'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Number of epochs to train for'
    )

    return parser.parse_args()


def load_imgs(path, percent_of_data, img_size):
    """Loads all images in a directory into one numpy array.

    Args:
        path (str): The path to a directory containing images to load in
        percent_of_data (float): The percentage of images from the directory to
        load in.
        img_size (int): The number of pixels in height and width to resize each
        image to before loading it into the numpy array.

    Returns:
        ndarray: A 4D numpy array of shape 
        (num_imgs_in_directory*percent_of_data/100, img_size, img_size, 3) 
        representing the randomly selected sample of reshaped images with 
        three color channels.
    """
    files = listdir(path)
    shuffle(files)
    files = files[:int(len(files)*percent_of_data/100)]
    imgs = []
    for fp in tqdm(files, total=len(files)):
        # uses this method for concurrency memory issue
        temp = Image.open(f'{path}/{fp}')
        temp = temp.resize((img_size, img_size), Image.LANCZOS)
        keep = np.array(temp.copy())
        imgs.append(np.array(keep))
        temp.close()
    return np.array(imgs)


def load_data(data_type, data_path, percent_of_data, img_size):
    """Loads the real and fake data for a given type of data and generates 
    associated labels.

    Args:
        data_type (str): 'train', 'valid', or 'test'.
        data_path (str): Path to the data parent directory.
        percent_of_data (float): The percent of images to load.
        img_size (int): The number of pixels in height and width to resize each
        image to.

    Returns:
        dictionary: Given some num_imgs = 
        (100000 if data_type=='train' else 20000)*percent_of_data/100,
        a dictionary with data of images of shape 
        (num_imgs, img_size, img_size, 30) and labels of shape (num_imgs, 2).
    """
    path = f'{data_path}/{data_type}/'
    (reals, fakes) = [load_imgs(path + t, percent_of_data, img_size)
                      for t in ['real', 'fake']]
    return {
        'data': np.vstack([reals, fakes]),
        'labels': to_categorical(np.hstack([
            np.ones((reals.shape[0])),
            np.zeros((fakes.shape[0]))
        ]))
    }


def main():
    """Runs model using command line arguments.
    """
    args = parse_args()
    check_paths(args.data_path)
    data = {t: load_data(t, args.data_path, args.percent_of_data, args.img_size)
            for t in ['train', 'valid', 'test']}
    model = create_model(args.truncate_block_num,
                         args.img_size, args.batch_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(data['train']['data'], data['train']['labels'],
              batch_size=args.batch_size, epochs=args.batch_size)
    # print(model.evaluate(data['test']['data'], data['test']['labels']))


if __name__ == "__main__":
    main()
