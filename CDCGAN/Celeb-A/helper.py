import numpy as np
import math
from PIL import Image

def get_image(image_path, width, height, mode):
    """
    Read image from image_path
    param image_path: Path of image
    :param width: Width of image
    :param height: Height of image
    :param mode: Mode of image
    :return: Image data
    """

    image = Image.open(image_path)

    if image.size != (width, height):  # HACK - Check if image is from the CELEBA dataset
        # Remove most pixels that aren't part of a face
        face_width = face_height = 108
        j = (image.size[0] - face_width) // 2
        i = (image.size[1] - face_height) // 2
        image = image.crop([j, i, j + face_width, i + face_height])
        image = image.resize([width, height], Image.BILINEAR)

    return np.array(image.convert(mode))


def get_batch(image_files, width, height, mode):
    data_batch = np.array([get_image(sample_file, width, height, mode) for sample_file in image_files]).astype(np.float32)

    # Make sure the images are in 4 dimensions
    if len(data_batch.shape) < 4:
        data_batch = data_batch.reshape(data_batch.shape + (1,))

    return data_batch

def images_square_grid(images, mode):
    """
    Save images as a square grid
    :param images: Images to be used for the grid
    :param mode: The mode to use for images
    :return: Image of images in a square grid
    """
    # Get maximum size for square grid of images
    save_size = math.floor(np.sqrt(images.shape[0]))

    # Scale to 0-255
    images = (((images - images.min()) * 255) / (images.max() - images.min())).astype(np.uint8)

    # Put images in a square arrangement
    images_in_square = np.reshape(
            images[:save_size*save_size],
            (save_size, save_size, images.shape[1], images.shape[2], images.shape[3]))
    if mode == 'L':
        images_in_square = np.squeeze(images_in_square, 4)

    # Combine images to grid image
    new_im = Image.new(mode, (images.shape[1] * save_size, images.shape[2] * save_size))
    for col_i, col_images in enumerate(images_in_square):
        for image_i, image in enumerate(col_images):
            im = Image.fromarray(image, mode)
            new_im.paste(im, (col_i * images.shape[1], image_i * images.shape[2]))

    return new_im

class Dataset(object):
    """
    Dataset
    """
    def __init__(self, data_files):
        """
        Initalize the class
        :param data_files: List of files in the database
        """
        IMAGE_WIDTH = 64 # 28
        IMAGE_HEIGHT = 64 # 28
        self.image_mode = 'RGB'
        image_channels = 3

        self.data_files = data_files
        self.shape = len(data_files), IMAGE_WIDTH, IMAGE_HEIGHT, image_channels

        self.num_examples = self.shape[0]
        self.batch_index = 0

    def get_batches(self, batch_size):
        """
        Generate batches
        :param batch_size: Batch Size
        :return: Batches of data
        """
        IMAGE_MAX_VALUE = 255

        current_index = 0
        while current_index + batch_size <= self.shape[0]:
            data_batch = get_batch(
                self.data_files[current_index:current_index + batch_size],
                *self.shape[1:3],
                self.image_mode)

            current_index += batch_size

            # make range between -1 ~ 1
            yield (data_batch / IMAGE_MAX_VALUE - 0.5) * 2.0

    def get_next_batch(self, batch_size):
        IMAGE_MAX_VALUE = 255

        if (self.batch_index + batch_size) > self.num_examples:
            self.batch_index = 0

        data_batch = get_batch( self.data_files[self.batch_index:self.batch_index + batch_size],
                                *self.shape[1:3],
                                self.image_mode)
        self.batch_index += batch_size

        return (data_batch / IMAGE_MAX_VALUE - 0.5) * 2.0

class AttrParserCelebA(object):
    def __init__(self, attr_fn, attr_name):
        '''
        :param attr_fn: attribute text file name
        :param attr_name: name of the attribute
        '''
        # get number of data & atrtibute types
        with open(attr_fn, 'r') as f:
            # parse number of data
            first_line = f.readline()
            self.n_data = int(first_line)

            # parse each attribute names & size
            second_line = f.readline()
            self.attr_names = second_line.split()
            self.n_attr = len(self.attr_names)

        # load whole text file as in np ndarray(2d matrix: n_data x n_attr)
        #self.attr_data = np.loadtxt(attr_fn, dtype=int, skiprows=2, usecols=range(1, self.n_attr + 1))
        self.attr_data = np.loadtxt(attr_fn, dtype=int, skiprows=2, usecols=range(1, self.n_attr + 1))

        # data contains -1: False, 1: True
        # convert that to 0 & 1
        self.attr_data = (self.attr_data + 1) // 2

        # grab attribute speific labels first
        raw_attr_data = self.get_labels_by_attr_name(attr_name)

        # one-hot encode (two class: True or False)
        self.to_batch_data = np.zeros( (self.n_data, 2) )
        self.to_batch_data[np.arange(self.n_data), raw_attr_data] = 1

        # initialize batch index
        self.batch_index = 0

    def get_next_batch(self, batch_size):
        """
        Generate batches
        :param batch_size: Batch Size
        :return: Batches of data
        """
        if (self.batch_index + batch_size) > self.n_data:
            self.batch_index = 0

        data = self.to_batch_data[self.batch_index:self.batch_index + batch_size,:]
        self.batch_index += batch_size
            
        return data

    def get_labels_by_attr_name(self, attr_name):
        """
        Get all data via attribute name
        :param attr_name: name of the attribute
        :return: data of given attribute name
        """
        # get index of attribute
        try:
            attr_index = self.attr_names.index(attr_name)
        except:
            print('unidentified attribute name...!!!')

        return self.attr_data[:, attr_index]