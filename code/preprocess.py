import numpy as np
import os, sys
import cv2
from PIL import Image
from tqdm import tqdm
    
def check_paths(parent_path):
    ######################### assert that the paths are there #########################
    absolute_path = str(os.getcwd()) + '/' + parent_path
    train_real_path, train_fake_path = absolute_path + "/train/real", absolute_path + "/train/fake"
    test_real_path, test_fake_path = absolute_path + "/test/real", absolute_path + "/test/fake"
    valid_real_path, valid_fake_path = absolute_path + "/valid/real", absolute_path + "/valid/fake"

    if (not os.path.exists(train_real_path)) or (not os.path.exists(train_fake_path)):
        raise NotImplementedError(f'Please make sure that /{parent_path}/train/real and /{parent_path}/train/fake exist')

    if (not os.path.exists(test_real_path)) or (not os.path.exists(test_fake_path)):
        raise NotImplementedError(f'Please make sure that /{parent_path}/test/real and /{parent_path}/test/fake exist')

    if (not os.path.exists(valid_real_path)) or (not os.path.exists(valid_fake_path)):
        raise NotImplementedError(f'Please make sure that /{parent_path}/valid/real and /{parent_path}/valid/fake exist')

    # alright. now create folders to put preprocessed files if they are not there already
    train_preprocess_path = absolute_path + "/train/preprocess"
    if not os.path.exists(train_preprocess_path):
        os.mkdir(train_preprocess_path, mode=0o775)

    train_real_preprocess_path, train_fake_preprocess_path = train_preprocess_path + "/real", \
                                                                train_preprocess_path + "/fake"
    
    if not os.path.exists(train_real_preprocess_path):
        os.mkdir(train_real_preprocess_path, mode=0o775)

    if not os.path.exists(train_fake_preprocess_path):
        os.mkdir(train_fake_preprocess_path, mode=0o775)
    

def preprocessing(parent_path):
    #    In order to minimize differences between the real and fake images and reduce
    #    artificats that could impact accurary the following steps are need (Chai et al, pg 22):
    #    Real Images:
    #         (1) Pass real images through generator data transformed
    #         (2) All images are resized to the same (128 pc) before saving to PNG format
    #         (3) Image would then be resize to classifiers native resolution, mean centering would then be performed
    #     Fake Images
    #         Step 1 would be replace with sampling and renormalizing the output from the generator
    
    absolute_path = str(os.getcwd()) + '/' + parent_path
    train_real_path, train_fake_path = absolute_path + "/train/real", absolute_path + "/train/fake"
    train_preprocess_path = absolute_path + "/train/preprocess"
    train_real_preprocess_path, train_fake_preprocess_path = train_preprocess_path + "/real", \
                                                                train_preprocess_path + "/fake"


    train_real_files, train_fake_files = os.listdir(train_real_path), os.listdir(train_fake_path)
    train_real_preprocess_files, train_fake_preprocess_files = os.listdir(train_real_preprocess_path),\
                                                                 os.listdir(train_fake_preprocess_path)

    MARGIN_OF_ERROR_IN_NUM_FILES = 10 # real bad engineering practice but im just tryna graduate
    
    if abs(len(train_real_files) - len(train_real_preprocess_files)) > MARGIN_OF_ERROR_IN_NUM_FILES:
        # dealing with real files
        for fp in tqdm(train_real_files, total=len(train_real_files)):
            filepath = fp.split(".")
            im = Image.open(train_real_path + "/{}".format(fp))
            im = im.resize((128, 128), Image.LANCZOS)
            png_path = train_real_preprocess_path + "/{}.png".format(filepath[0])
            im.save(png_path)

    if abs(len(train_fake_files) - len(train_fake_preprocess_files)) > MARGIN_OF_ERROR_IN_NUM_FILES:
        # dealing with fake files
        for fp in tqdm(train_fake_files, total=len(train_fake_files)):
            filepath = fp.split(".")
            im = Image.open(train_fake_path + "/{}".format(fp))
            im = im.resize((128,128), Image.LANCZOS)
            png_path = train_fake_preprocess_path + "/{}.png".format(filepath[0])
            im.save(png_path)


    
    
    