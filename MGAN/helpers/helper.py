import numpy as np
import os
import glob
from scipy.misc import toimage

import cv2


def preprocess_for_saving_image(im):
    if im.shape[0] == 1:
        im = np.squeeze(im, axis=0)

    # Scale to 0-255
    # im = (((im - im.min()) * 255) / (im.max() - im.min())).astype(np.uint8)
    im = (((im + 1.0) * 255) / 2.0).astype(np.uint8)

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
    def __init__(self, input_dir, direction='AtoB', do_flip=True, is_test=False):
        if not os.path.exists(input_dir):
            raise Exception('input directory does not exists!!')

        # search for images(*.jpg or *.png)
        jpg_files = glob.glob(os.path.join(input_dir, '*.jpg'))
        png_files = glob.glob(os.path.join(input_dir, '*.png'))
        self.image_files = jpg_files + png_files

        if len(self.image_files) == 0:
            raise Exception('input directory does not contain any images!!')

        # set class attributes
        self.n_images = len(self.image_files)
        self.direction = direction
        self.scale_to = 580
        self.do_flip = do_flip
        self.batch_index = 0
        self.image_max_value = 255
        self.crop_size = 512
        self.is_test = is_test
        self.n_rand_pt = self.crop_size // 5
        self.rand_roi_size = self.crop_size // 16

        # synchronize seed for image operations so that we do the same operations to both
        # input and output images
        seed = np.random.randint(0, 2 ** 31 - 1)
        self.prng = np.random.RandomState(seed)

    def get_next_batch(self, batch_size):
        if (self.batch_index + batch_size) > self.n_images:
            self.batch_index = 0

        current_files = self.image_files[self.batch_index:self.batch_index + batch_size]
        splitted = self.load_images(current_files)

        self.batch_index += batch_size

        return splitted

    def get_image_by_index(self, index):
        if index >= self.n_images:
            index = 0
        current_file = [self.image_files[index]]
        splitted = self.load_images(current_file)
        return splitted

    def load_images(self, files):
        def normalize(img):
            # normalize input [0 ~ 255] ==> [-1 ~ 1]
            img = img.astype(np.float32)
            img = (img / self.image_max_value - 0.5) * 2
            return img

        splitted = []
        for img_fn in files:
            # open with opencv
            img = cv2.imread(img_fn)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, channels = img.shape

            # crop image
            a_image = img[:, :width // 2, :]
            b_image = img[:, width // 2:, :]

            # apply random flip, resize, random crop
            if self.is_test is False:
                random_val_flip = self.prng.uniform(0, 1)
                offset_h = np.floor(self.prng.uniform(0, self.scale_to - self.crop_size + 1)).astype(np.int32)
                offset_w = np.floor(self.prng.uniform(0, self.scale_to - self.crop_size + 1)).astype(np.int32)
                a_image = self.transform(a_image, random_val_flip, offset_h, offset_w)
                b_image = self.transform(b_image, random_val_flip, offset_h, offset_w)

            if self.direction == 'AtoB':
                sketch, color = [a_image, b_image]
            elif self.direction == 'BtoA':
                sketch, color = [b_image, a_image]
            else:
                raise Exception('Invalid direction')

            color_hint = self.spread_random_blur(color)

            sketch = normalize(sketch)
            color = normalize(color)
            color_hint = normalize(color_hint)

            splitted.append((sketch, color, color_hint))
        return splitted


    def transform(self, img, random_val, offset_h, offset_w):
        r = img

        # perform random flip
        if self.do_flip and random_val >= 0.5:
            r = cv2.flip(r, flipCode=1)

        # resize to add random jitter
        r = cv2.resize(r, (self.scale_to, self.scale_to))

        # randomly crop back to original size
        if self.scale_to > self.crop_size:
            r = r[offset_h:offset_h + self.crop_size, offset_w:offset_w + self.crop_size, :]
        elif self.scale_to < self.crop_size:
            raise Exception("scale size cannot be less than crop size")

        return r

    def spread_random_blur(self, img_color):
        loc_x = np.random.randint(0, self.crop_size, self.n_rand_pt)
        loc_y = np.random.randint(0, self.crop_size, self.n_rand_pt)

        roi_calc = self.rand_roi_size // 2
        max_boundary = self.crop_size - 1
        mask = np.zeros_like(img_color[:,:,0], dtype=np.uint8)
        np.squeeze(mask)
        for x, y in zip(loc_x, loc_y):
            x_start = max(x - roi_calc, 0)
            x_end = min(x + roi_calc, max_boundary)

            y_start = max(y - roi_calc, 0)
            y_end = min(y + roi_calc, max_boundary)

            mask[y_start:y_end, x_start:x_end] = 255

        background = np.ones_like(img_color, dtype=np.uint8) * 255
        mask_inv = cv2.bitwise_not(mask)
        color_fg = cv2.bitwise_and(img_color, img_color, mask=mask)
        color_bg = cv2.bitwise_and(background, background, mask=mask_inv)
        added = cv2.add(color_bg, color_fg)

        # perform gaussian blur
        blurred = cv2.GaussianBlur(added, (55, 55), 0)

        return blurred

    # def spread_random_blur(self, img_sketch, img_color):
    #     # pick random places to extract color
    #     mask = np.random.choice(2, (self.crop_size, self.crop_size), p=[0.01, 0.99]).astype(bool)
    #     mask = np.expand_dims(mask, axis=2)
    #     mask = np.repeat(mask, repeats=3, axis=2)
    #     masked = np.ma.MaskedArray(img_color, mask, fill_value=255)
    #     picked = masked.filled()
    #
    #     # perform gaussian blur
    #     blurred = cv2.GaussianBlur(picked, (5, 5), 0)
    #
    #     # add two images
    #     added = cv2.addWeighted(img_sketch, 0.5, blurred, 0.5, 0.0)
    #
    #     # normalize input [0 ~ 255] ==> [-1 ~ 1]
    #     added = added.astype(np.float32)
    #     added = (added / self.image_max_value - 0.5) * 2
    #
    #     return added


def create_test_set():
    train_input_dir = 'd:/db/getchu/merged_512/'
    # prepare dataset
    train_dataset = Dataset(train_input_dir, direction='BtoA', is_test=True)

    def get_image(loader, index):
        batch_images_tuple = loader.get_image_by_index(index)
        a = [x for x, y, z in batch_images_tuple]
        b = [y for x, y, z in batch_images_tuple]
        c = [z for x, y, z in batch_images_tuple]
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        return a, b, c

    save_dir = './test'
    for ii in range(6):
        sketch, color, hint = get_image(train_dataset, ii)
        sketch = preprocess_for_saving_image(sketch)
        color = preprocess_for_saving_image(color)
        hint = preprocess_for_saving_image(hint)

        sketch_fn = os.path.join(save_dir, 'sketch-{:02d}.png'.format(ii))
        color_fn = os.path.join(save_dir, 'color-{:02d}.png'.format(ii))
        hint_fn = os.path.join(save_dir, 'hint-{:02d}.png'.format(ii))

        toimage(sketch, mode='RGB').save(sketch_fn)
        toimage(color, mode='RGB').save(color_fn)
        toimage(hint, mode='RGB').save(hint_fn)

    return