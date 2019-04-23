import tensorflow as tf
import tensorflow_datasets as tfds


def preprocess_fn(features):
    # [0 ~ 255] -> [0.0 ~ 1.0]
    features['image'] = tf.image.convert_image_dtype(features['image'], dtype=tf.float32)

    # [0.0 ~ 1.0] -> [-1.0 ~ 1.0]
    features['image'] = (features['image'] - 0.5) * 2.0
    return features


def get_mnist_by_name(batch_size, name='mnist'):
    # will return [28, 28, 1] uint8 (0~255)
    dataset = tfds.load(name=name, split=tfds.Split.ALL)
    dataset = dataset.shuffle(70000 + 1)

    dataset = dataset.map(lambda x: preprocess_fn(x))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def main():
    # test mnist loaders
    batch_size = 100
    epochs = 1
    name = 'mnist'
    # name = 'fashion_mnist'
    dataset = get_mnist_by_name(batch_size, name)

    n_images = 0
    with tf.Session() as sess:
        for e in range(epochs):
            iterator = dataset.make_one_shot_iterator()
            next_elem = iterator.get_next()
            while True:
                try:
                    elem = sess.run(next_elem)
                    images = elem['image']
                    labels = elem['label']
                    n_images += images.shape[0]
                    # print()
                except tf.errors.OutOfRangeError:
                    print('End of dataset')
                    break

    print('{}'.format(n_images))
    return


if __name__ == '__main__':
    main()
