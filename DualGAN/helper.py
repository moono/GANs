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

def save_result(image_fn,
                real_image_u, g_image_u_to_v, g_image_u_to_v_to_u,
                real_image_v, g_image_v_to_u, g_image_v_to_u_to_v):
    im_0 = preprocess_for_saving_image(real_image_u)
    im_1 = preprocess_for_saving_image(g_image_u_to_v)
    im_2 = preprocess_for_saving_image(g_image_u_to_v_to_u)
    im_3 = preprocess_for_saving_image(real_image_v)
    im_4 = preprocess_for_saving_image(g_image_v_to_u)
    im_5 = preprocess_for_saving_image(g_image_v_to_u_to_v)

    concat_row_0 = np.concatenate((im_0, im_1, im_2), axis=1)
    concat_row_1 = np.concatenate((im_3, im_4, im_5), axis=1)
    concated = np.concatenate((concat_row_0, concat_row_1), axis=0)

    if concated.shape[2] == 1:
        reshaped = np.squeeze(concated, axis=2)
        toimage(reshaped, mode='L').save(image_fn)
    else:
        toimage(concated, mode='RGB').save(image_fn)



# class for loading images
class Dataset(object):
    def __init__(self, input_dir_u, input_dir_v, fn_ext, im_size, im_channel_u, im_channel_v, do_flip, do_shuffle):
        if not os.path.exists(input_dir_u) or not os.path.exists(input_dir_v):
            raise Exception('input directory does not exists!!')

        # search for images
        self.image_files_u = glob.glob(os.path.join(input_dir_u, '*.{:s}'.format(fn_ext)))
        self.image_files_v = glob.glob(os.path.join(input_dir_v, '*.{:s}'.format(fn_ext)))
        if len(self.image_files_u) == 0 or len(self.image_files_v) == 0:
            raise Exception('input directory does not contain any images!!')

        # shuffle image files
        if do_shuffle:
            np.random.shuffle(self.image_files_u)
            np.random.shuffle(self.image_files_v)

        self.n_images = len(self.image_files_u) if len(self.image_files_u) <= len(self.image_files_v) else len(self.image_files_v)
        self.batch_index = 0
        self.resize_to = im_size
        self.color_mode_u = 'L' if im_channel_u == 1 else 'RGB'
        self.color_mode_v = 'L' if im_channel_v == 1 else 'RGB'
        self.do_flip = do_flip
        self.image_max_value = 255
        # self.prng = np.random.RandomState(777)

    def get_image_by_index(self, index):
        if index >= self.n_images:
            index = 0

        fn_u = [self.image_files_u[index]]
        fn_v = [self.image_files_v[index]]
        image_u = self.load_image(fn_u, self.color_mode_u)
        image_v = self.load_image(fn_v, self.color_mode_v)
        return image_u, image_v

    def get_next_batch(self, batch_size):
        if (self.batch_index + batch_size) > self.n_images:
            self.batch_index = 0

        batch_files_u = self.image_files_u[self.batch_index:self.batch_index + batch_size]
        batch_files_v = self.image_files_v[self.batch_index:self.batch_index + batch_size]

        images_u = self.load_image(batch_files_u, self.color_mode_u)
        images_v = self.load_image(batch_files_v, self.color_mode_v)

        return images_u, images_v

    def load_image(self, fn_list, color_mode):
        images = []
        for fn in fn_list:
            # open images with PIL
            im = Image.open(fn)
            im = np.array(im.convert(color_mode)).astype(np.float32)

            # resize
            if im.shape[0] is not self.resize_to or im.shape[1] is not self.resize_to:
                im = imresize(im, [self.resize_to, self.resize_to])

            # perform flip if needed
            # random_val = self.prng.uniform(0, 1)
            random_val = np.random.random()
            if self.do_flip and random_val > 0.5:
                im = np.flip(im, axis=1)

            # normalize input [0 ~ 255] ==> [-1 ~ 1]
            im = (im / self.image_max_value - 0.5) * 2

            # make 3 dimensional for single channel image
            if len(im.shape) < 3:
                im = np.expand_dims(im, axis=2)

            images.append(im)
        images = np.array(images)

        return images