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

        # if the image names are numbers, sort by the value rather than asciibetically
        # having sorted inputs means that the outputs are sorted in test mode
        if all(get_name(path).isdigit() for path in self.image_files):
            self.image_files = sorted(self.image_files, key=lambda path: int(get_name(path)))
        else:
            self.image_files = sorted(self.image_files)

        # set class attributes
        self.n_images = len(self.image_files)
        self.convert_to_lab_color = convert_to_lab_color
        self.direction = direction
        self.scale_to = 286
        self.do_flip = do_flip
        self.batch_index = 0
        self.image_max_value = 255
        self.crop_size = 256
        self.is_test = is_test

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
        splitted = []
        for im in files:
            # open images with PIL
            im = Image.open(im)

            if self.convert_to_lab_color:
                raise Exception('Lab color conversion: Not implemented yet!!')
            else:
                # convert to np array
                im = np.array(im.convert('RGB')).astype(np.float32)
                width = im.shape[1]  # [height, width, channels]
                a_image = im[:, :width // 2, :]
                b_image = im[:, width // 2:, :]

            if self.is_test:
                # testing examples don't need random flip & random cropping
                a_image = self.transform_testing(a_image)
                b_image = self.transform_testing(b_image)
            else:
                # at training...
                # apply random flip, resize, random crop
                random_val_flip = self.prng.uniform(0, 1)
                offset_h = np.floor(self.prng.uniform(0, self.scale_to - self.crop_size + 1)).astype(np.int32)
                offset_w = np.floor(self.prng.uniform(0, self.scale_to - self.crop_size + 1)).astype(np.int32)
                a_image = self.transform(a_image, random_val_flip, offset_h, offset_w)
                b_image = self.transform(b_image, random_val_flip, offset_h, offset_w)

            if self.direction == 'AtoB':
                inputs, targets = [a_image, b_image]
            elif self.direction == 'BtoA':
                inputs, targets = [b_image, a_image]
            else:
                raise Exception('Invalid direction')

            splitted.append((inputs, targets))
        return splitted

    def transform(self, im, random_val, offset_h, offset_w):
        r = im

        # perform random flip
        if self.do_flip and random_val >= 0.5:
            r = np.flip(r, axis=1)

        # resize to add random jitter
        r = imresize(r, [self.scale_to, self.scale_to])

        # randomly crop back to original size
        if self.scale_to > self.crop_size:
            r = r[offset_h:offset_h + self.crop_size, offset_w:offset_w + self.crop_size, :]
        elif self.scale_to < self.crop_size:
            raise Exception("scale size cannot be less than crop size")

        # normalize input [0 ~ 255] ==> [-1 ~ 1]
        r = (r / self.image_max_value - 0.5) * 2
        return r

    def transform_testing(self, im):
        r = im

        if r.shape[0] is not self.crop_size or r.shape[1] is not self.crop_size:
            r = imresize(r, [self.crop_size, self.crop_size])

        # normalize input [0 ~ 255] ==> [-1 ~ 1]
        r = (r / self.image_max_value - 0.5) * 2
        return r
