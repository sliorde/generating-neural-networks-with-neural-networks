import gzip
import numpy as np
import sys

sys.path.append('../')

from tensorflow.examples.tutorials.mnist import input_data
from params import GeneralParameters

general_params = GeneralParameters()


class MNIST(object):
    def __int__(self):
        self.nb_classes = general_params.number_of_categories
        self.name = self.__class__.__name__.lower()

    def load_data(self):
        mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)

        train_images = np.reshape(mnist.train.images, (-1, general_params.image_height, general_params.image_width, general_params.number_of_channels))
        train_labels =  np.reshape(mnist.train.labels, (-1, general_params.number_of_categories))

        validation_images = np.reshape(mnist.validation.images,(-1, general_params.image_height, general_params.image_width, general_params.number_of_channels))
        validation_labels = np.reshape(mnist.validation.labels, (-1, general_params.number_of_categories))

        test_images = np.reshape(mnist.test.images,(-1, general_params.image_height, general_params.image_width, general_params.number_of_channels))
        test_labels = np.reshape(mnist.test.labels, (-1, general_params.number_of_categories))

        return [train_images, train_labels], [validation_images, validation_labels], [test_images, test_labels]

    def images(self, n=None):
        return self.load_data()
