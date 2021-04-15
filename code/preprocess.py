import numpy as np
import dill as pickle
import dnnlib
import dnnlib.tflib as tflib

def generator() :
    # Retrieve generator function for stylegan2-cache
    generator_url = 'gdrive:networks/stylegan2-ffhq-config-a.pkl'
    stream = dnnlib.util.open_url(generator_url , cache_dir = 'data/real/stylegan2-cache')
    
    tflib.init_tf()
    with stream :
        G, D, Gs = pickle.load(stream, encoding = 'latin1')

    return G, D, Gs

def preprocessing(filepath) :
#    In order to minimize differences between the real and fake images and reduce
#    artificats that could impact accurary the following steps are need (Chai et al, pg 22):
#    Real Images:
#         (1) Pass real images through generator data transformed
#         (2) All images are resized to the dame (128 pc) before saving to PNG format
#         (3) Image would then be resize to classifiers native resolution, mean centering would then be performed
#     Fake Images
#         Step 1 would be replace with sampling and renormalizing the output from the generator
    
    return




    
    
    