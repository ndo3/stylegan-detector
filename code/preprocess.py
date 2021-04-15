import numpy as np
import os
from PIL import Image
    
def preprocessing(filepath) :
#    In order to minimize differences between the real and fake images and reduce
#    artificats that could impact accurary the following steps are need (Chai et al, pg 22):
#    Real Images:
#         (1) Pass real images through generator data transformed
#         (2) All images are resized to the same (128 pc) before saving to PNG format
#         (3) Image would then be resize to classifiers native resolution, mean centering would then be performed
#     Fake Images
#         Step 1 would be replace with sampling and renormalizing the output from the generator
    
    
    real_files = os.listdir("stylegan-detector/data/train/real")
    fake_files = os.listdir("stylegan-detector/data/train/real")
    
    for fp in real_files :
        filepath = fp.split(".")
        im = Image.open(fp)
        im = np.transpose(im, (1,2,0))
        im = im.resize((128,128), Image.LANCZOS)
        # im.save(os.path.join("/data/train/real/", filepath[0], ".png"))
    
    for fp in fake_files :
        filepath = fp.split(".")
        im = Image.open(fp)
        im = np.clip(np.rint((im + 1.0) / 2.0 * 255.0), 0.0,
                         255.0).astype(np.uint8) # [-1,1] => [0,255]
        im = im.transpose(0, 2, 3, 1) # NCHW => NHWC
        im = im.resize((128,128), Image.LANCZOS)
        # im.save(os.path.join("/data/train/fake/", filepath[0], ".png"))

    return




    
    
    