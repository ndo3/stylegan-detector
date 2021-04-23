import numpy as np
import os
import sys
import cv2
from PIL import Image
from tqdm import tqdm


def check_paths(parent_path):
    """Checks that all the paths for the data setup are as expected. 

    Args:
        parent_path (str): The relative path from the current directory 
        containing the data.

    Raises:
        NotImplementedError: If one of the paths is missing
    """
    absolute_path = str(os.getcwd()) + '/' + parent_path
    for path_type in ["train", "test", "valid"]:
        real_path = absolute_path + f"/{path_type}/real"
        fake_path = absolute_path + f"/{path_type}/fake"

        if (not os.path.exists(real_path)) or (not os.path.exists(fake_path)):
            raise NotImplementedError(
                f'Please make sure that /{parent_path}/train/real and /{parent_path}/{path_type}/fake exist'
            )


def preprocessing(parent_path, percent_of_data=100):
    """Creates a preprocessed subdirectory in the data files and processes each
    image into a smaller image. IMPORTANT NOTE: we ended up not using this
    function because it took a lot of space to save preprocessed images
    separately: however, these could be useful in the future for running the
    model multiple times to reduce redundant work.

    Args:
        parent_path (str): The relative path from the current directory 
        containing the data.
        percent_of_data (float): The percent of each data directory to 
        preprocess; useful for running the model on a subset.
    """
    absolute_path = str(os.getcwd()) + '/' + parent_path
    MARGIN_OF_ERROR_IN_NUM_FILES = 10
    for data_type in ["train", "test", "valid"]:
        real_path = absolute_path + f"/{data_type}/real"
        fake_path = absolute_path + f"/{data_type}/fake"
        preprocess_path = absolute_path + f"/{data_type}/preprocess"

        if not os.path.exists(preprocess_path):
            os.mkdir(preprocess_path, mode=0o775)

        real_preprocess_path = preprocess_path + "/real"
        fake_preprocess_path = preprocess_path + "/fake"

        if not os.path.exists(real_preprocess_path):
            os.mkdir(real_preprocess_path, mode=0o775)

        if not os.path.exists(fake_preprocess_path):
            os.mkdir(fake_preprocess_path, mode=0o775)

        real_files, fake_files = os.listdir(real_path), os.listdir(fake_path)
        real_preprocess_files = os.listdir(real_preprocess_path)
        fake_preprocess_files = os.listdir(fake_preprocess_path)

        correct_prefixes = [real_preprocess_path, fake_preprocess_path]
        for i, filelist in enumerate([real_preprocess_files, fake_preprocess_files]):
            for fp in tqdm(filelist, total=len(filelist)):
                to_remove_filepath = correct_prefixes[i] + f"/{fp}"
                os.remove(to_remove_filepath)

        real_preprocess_files = os.listdir(real_preprocess_path)
        fake_preprocess_files = os.listdir(fake_preprocess_path)

        assert len(real_preprocess_files) == 0
        assert len(fake_preprocess_files) == 0

        real_files = real_files[:int(len(real_files)*percent_of_data/100)]
        fake_files = fake_files[:int(len(fake_files)*percent_of_data/100)]

        if abs(len(real_files) - len(real_preprocess_files)) > MARGIN_OF_ERROR_IN_NUM_FILES:
            # dealing with real files
            for fp in tqdm(real_files, total=len(real_files)):
                filepath = fp.split(".")
                im = Image.open(real_path + "/{}".format(fp))
                im = im.resize((224, 224), Image.LANCZOS)
                png_path = real_preprocess_path + "/{}.png".format(filepath[0])
                im.save(png_path)

        if abs(len(fake_files) - len(fake_preprocess_files)) > MARGIN_OF_ERROR_IN_NUM_FILES:
            # dealing with fake files
            for fp in tqdm(fake_files, total=len(fake_files)):
                filepath = fp.split(".")
                im = Image.open(fake_path + "/{}".format(fp))
                im = im.resize((224, 224), Image.LANCZOS)
                png_path = fake_preprocess_path + "/{}.png".format(filepath[0])
                im.save(png_path)
