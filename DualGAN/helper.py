import numpy as np
import os
import glob
from PIL import Image
from scipy.misc import imresize, toimage


def preprocess_for_saving_image(im):
    if im.shape[0] == 1:
        im = np.squeeze(im, axis=0)

    # Scale to 0-255
    im = (((im - im.min()) * 255) / (im.max() - im.min())).astype(np.uint8)

    return im

def save_result(image_fn, gen_image, input_image=None, target_image=None):
    image_1 = preprocess_for_saving_image(gen_image)
    concated_image = image_1
    if input_image is not None:
        image_2 = preprocess_for_saving_image(input_image)
        concated_image = np.concatenate((concated_image, image_2), axis=1)
    if target_image is not None:
        image_3 = preprocess_for_saving_image(target_image)
        concated_image = np.concatenate((concated_image, image_3), axis=1)

    toimage(concated_image, mode='RGB').save(image_fn)

# class for loading images & split image of the form [A|B] ==> (A, B)
class Dataset(object):
    def __init__(self, input_dir, convert_to_lab_color=False, direction='AtoB', do_flip=False, is_test=False):
        if not os.path.exists(input_dir):
            raise Exception('input directory does not exists!!')

        # search for images(*.jpg or *.png)
        self.image_files = glob.glob(os.path.join(input_dir, '*.jpg'))
        if len(self.image_files) == 0:
            self.image_files = glob.glob(os.path.join(input_dir, '*.png'))

        if len(self.image_files) == 0:
            raise Exception('input directory does not contain any images!!')

        def get_name(path):
            name, _ = os.path.splitext(os.path.basename(path))
            return name

