# program to resize all train images

import os
import glob
from PIL import Image
from joblib import Parallel, delayed

in_dir = 'train_images/'
out_dir = 'train256/'
IMAGE_SIZE = 256

JPG_FILES = glob.glob(in_dir + '*.jpg')


def convert(img_file):
    im = Image.open(img_file)
    im.resize((IMAGE_SIZE, IMAGE_SIZE)).save(out_dir + os.path.basename(img_file), 'JPEG')


Parallel(n_jobs=-1, verbose=10)(delayed(convert)(f) for f in JPG_FILES)
