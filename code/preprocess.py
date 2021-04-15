import numpy as np
import os
import cv2
from PIL import Image
    
def preprocessing() :
#    In order to minimize differences between the real and fake images and reduce
#    artificats that could impact accurary the following steps are need (Chai et al, pg 22):
#    Real Images:
#         (1) Pass real images through generator data transformed
#         (2) All images are resized to the same (128 pc) before saving to PNG format
#         (3) Image would then be resize to classifiers native resolution, mean centering would then be performed
#     Fake Images
#         Step 1 would be replace with sampling and renormalizing the output from the generator
    
    absolute_path = "/Users/esmeraldamontas/Documents/Senior Year/Spring Semster/CSCI 1430/stylegan-detector/data/"

    real_files = os.listdir(absolute_path + "/train/real")
    fake_files = os.listdir(absolute_path + "/train/fake")
    
    new_size = (128, 128)
    for fp in real_files :
        filepath = fp.split(".")
        im = Image.open(absolute_path + "/train/real/" + fp)
        im = np.transpose(im, (1,2,0))
        im = Image.fromarray(im, 'RGB')
        im = im.resize((128, 128), Image.LANCZOS)
        png_path = absolute_path + "/train/preprocess/real/" + filepath[0] + ".png"
        im.save(png_path)
    
    for fp in fake_files :
        filepath = fp.split(".")
        im = Image.open(absolute_path + "/train/fake/" + fp)
        im = np.clip(np.rint((im + 1.0) / 2.0 * 255.0), 0.0,
                         255.0).astype(np.uint8) # [-1,1] => [0,255]
        im = im.transpose(0, 2, 3, 1) # NCHW => NHWC
        im = Image.fromarray(im, 'RGB')
        im = im.resize((128,128), Image.LANCZOS)
        png_path = absolute_path + "/train/preprocess/fake/" + filepath[0] + ".png"
        im.save(png_path)
    
    
if __name__ == '__main__':
    preprocessing()



    
    
    