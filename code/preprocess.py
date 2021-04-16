import numpy as np
import os, sys
import cv2
from PIL import Image
    
def check_paths():
    ######################### assert that the paths are there #########################
    absolute_path = str(os.getcwd()) + "/../data/"
    train_real_path, train_fake_path = absolute_path + "/train/real", absolute_path + "/train/fake"
    test_real_path, test_fake_path = absolute_path + "/test/real", absolute_path + "/test/fake"
    valid_real_path, valid_fake_path = absolute_path + "/valid/real", absolute_path + "/valid/fake"

    if (not os.path.exists(train_real_path)) or (not os.path.exists(train_fake_path)):
        raise NotImplementedError("Please make sure that /data/train/real and /data/train/fake exists")

    if (not os.path.exists(test_real_path)) or (not os.path.exists(test_fake_path)):
        raise NotImplementedError("Please make sure that /data/test/real and /data/test/fake exists")

    if (not os.path.exists(valid_real_path)) or (not os.path.exists(valid_fake_path)):
        raise NotImplementedError("Please make sure that /data/valid/real and /data/valid/fake exists")

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
    

def preprocessing() :
    #    In order to minimize differences between the real and fake images and reduce
    #    artificats that could impact accurary the following steps are need (Chai et al, pg 22):
    #    Real Images:
    #         (1) Pass real images through generator data transformed
    #         (2) All images are resized to the same (128 pc) before saving to PNG format
    #         (3) Image would then be resize to classifiers native resolution, mean centering would then be performed
    #     Fake Images
    #         Step 1 would be replace with sampling and renormalizing the output from the generator
    
    absolute_path = str(os.getcwd()) + "/../data/"
    train_real_path, train_fake_path = absolute_path + "/train/real", absolute_path + "/train/fake"
    train_real_preprocess_path, train_fake_preprocess_path = train_preprocess_path + "/real", \
                                                                train_preprocess_path + "/fake"

    train_real_files = os.listdir(train_real_path)
    train_fake_files = os.listdir(train_fake_path)
    
    # dealing with real files
    for fp in train_real_files :
        filepath = fp.split(".")
        im = Image.open(train_real_path + "/{}".format(fp))
        im = im.resize((128, 128), Image.LANCZOS)
        png_path = train_real_preprocess_path + "/{}.png".format(filepath[0])
        im.save(png_path)
    
    for fp in fake_files :
        filepath = fp.split(".")
        im = Image.open(train_fake_path + "/{}".format(fp))
        im = np.array(im)
        im = np.clip(np.rint((im + 1.0) / 2.0 * 255.0), 0.0,
                         255.0).astype(np.uint8) # [-1,1] => [0,255]
        im = np.transpose(im,(0, 2, 3, 1)) # NCHW => NHWC
        im = Image.fromarray(im, 'RGB')
        im = im.resize((128,128), Image.LANCZOS)
        png_path = train_fake_preprocess_path + "/{}.png".format(filepath[0])
        im.save(png_path)
    
    
if __name__ == '__main__':
    check_paths()
    preprocessing()



    
    
    