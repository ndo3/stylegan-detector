import numpy as np
import os, sys
import cv2
from PIL import Image
from tqdm import tqdm
    
def check_paths(parent_path):
    ######################### assert that the paths are there #########################
    absolute_path = str(os.getcwd()) + '/' + parent_path
    for path_type in ["train", "test", "valid"]:
        real_path, fake_path = absolute_path + f"/{path_type}/real", absolute_path + f"/{path_type}/fake"

        if (not os.path.exists(real_path)) or (not os.path.exists(fake_path)):
            raise NotImplementedError(f'Please make sure that /{parent_path}/train/real and /{parent_path}/{path_type}/fake exist')

    # # alright. now create folders to put preprocessed files if they are not there already
    # train_preprocess_path = absolute_path + "/train/preprocess"
    # if not os.path.exists(train_preprocess_path):
    #     os.mkdir(train_preprocess_path, mode=0o775)

    # train_real_preprocess_path, train_fake_preprocess_path = train_preprocess_path + "/real", \
    #                                                             train_preprocess_path + "/fake"
    
    # if not os.path.exists(train_real_preprocess_path):
    #     os.mkdir(train_real_preprocess_path, mode=0o775)

    # if not os.path.exists(train_fake_preprocess_path):
    #     os.mkdir(train_fake_preprocess_path, mode=0o775)
    

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
    MARGIN_OF_ERROR_IN_NUM_FILES = 10 # real bad engineering practice but im just tryna graduate
    for data_type in ["train", "test", "valid"]:
        real_path, fake_path = absolute_path + f"/{data_type}/real", absolute_path + f"/{data_type}/fake"
        preprocess_path = absolute_path + f"/{data_type}/preprocess"
        
        if not os.path.exists(preprocess_path):
            os.mkdir(preprocess_path, mode=0o775)

        real_preprocess_path, fake_preprocess_path = preprocess_path + "/real", \
                                                                    preprocess_path + "/fake"

        if not os.path.exists(real_preprocess_path):
            os.mkdir(real_preprocess_path, mode=0o775)
        
        if not os.path.exists(fake_preprocess_path):
            os.mkdir(fake_preprocess_path, mode=0o775)

        real_files, fake_files = os.listdir(real_path), os.listdir(fake_path)
        real_preprocess_files, fake_preprocess_files = os.listdir(real_preprocess_path),\
                                                                    os.listdir(fake_preprocess_path)
        
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
                im = im.resize((224,224), Image.LANCZOS)
                png_path = fake_preprocess_path + "/{}.png".format(filepath[0])
                im.save(png_path)