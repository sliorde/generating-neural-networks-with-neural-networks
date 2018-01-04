import numpy as np
import tensorflow as tf
import sys
import os

sys.path.append('../')

from mnf_mnist import MnfMnist
from mnist import MNIST
from params import ClassifierHyperParameters,GeneralParameters

general_params = GeneralParameters()
classifier_hparams = ClassifierHyperParameters()



class MnfNetwork():
    def __init__(self):

        mnist = MNIST()
        (xtrain, ytrain), _, _ = mnist.images()

        N, height, width, n_channels = xtrain.shape
        input_shape = [None, height, width, n_channels]

        model = MnfMnist(N, input_shape, 2, 2, True, learn_p=False,flow_dim_h=50, thres_var=0.5)
        x = tf.placeholder(tf.float32, [1, None, height, width, n_channels], name='x')
        xx = tf.reshape(x,[-1, height, width, n_channels])
        model.build_mnf_mnist(xx)

        self.z = model.z
        self.noise_size = model.noise_size

        self.x = x
        self.xx = xx
        self.model = model
        self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4 = model.layers[0].sample_weights() + model.layers[1].sample_weights() + model.layers[2].sample_weights() + model.layers[3].sample_weights()

        self.y_ = tf.placeholder(tf.float32, [None, general_params.number_of_categories], name='y_')
        self.y = model.predict(self.xx)
        self.probabilities = tf.expand_dims(tf.nn.softmax(self.y),0)
        self.prediction = tf.argmax(self.y, 1)
        self.correct_prediction = tf.equal(self.prediction, tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


        self.saver = tf.train.Saver()

        self.graph = tf.get_default_graph()

    def SampleInput(self, batch_size=None, task=None):
        if batch_size is None:
            batch_size = 1
        noise = np.random.normal(size=(batch_size, self.noise_size))
        return noise

    def GenerateWeights(self,sess:tf.Session,z=None,noise_batch_size=None):
        w1, b1, w2, b2, w3, b3, w4, b4 = [], [], [], [], [], [], [], []

        noise_batch_size = z.shape[0]

        for i in range(noise_batch_size):
            res = sess.run([self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4],feed_dict={self.z:z[[i],:]})
            w1.append(res[0])
            b1.append(np.reshape(res[1],[-1,classifier_hparams.layer1_size]))
            w2.append(res[2])
            b2.append(np.reshape(res[3],[-1,classifier_hparams.layer2_size]))
            w3.append(np.reshape(res[4],[-1,int(general_params.image_height / classifier_hparams.layer1_pool_size / classifier_hparams.layer2_pool_size),int(general_params.image_height / classifier_hparams.layer1_pool_size / classifier_hparams.layer2_pool_size),classifier_hparams.layer2_size,classifier_hparams.layer3_size]))
            b3.append(res[5])
            w4.append(res[6])
            b4.append(res[7])
        w1 = np.concatenate(w1)
        b1 = np.concatenate(b1)
        w2 = np.concatenate(w2)
        b2 = np.concatenate(b2)
        w3 = np.concatenate(w3)
        b3 = np.concatenate(b3)
        w4 = np.concatenate(w4)
        b4 = np.concatenate(b4)

        return self.Normalize(w1, b1, w2, b2, w3, b3, w4, b4)

    def Predict(self, sess: tf.Session, x, z = None, noise_batch_size = None,step_size=None):
        if x.ndim==5:
            x = x[0,:,:,:,:]
        noise_batch_size = z.shape[0]
        probs,preds = [],[]
        for i in range(noise_batch_size):
            res = sess.run([self.probabilities, self.prediction], feed_dict={self.xx:x, self.z:z[[i],:]})
            probs.append(np.expand_dims(res[0],0))
            preds.append(np.expand_dims(res[1],0))
        return np.concatenate(preds),np.squeeze(np.concatenate(probs),1)

    def GetAccuracy(self, sess: tf.Session, x, y, z, step_size=None):
        if x.ndim==5:
            x = x[0,:,:,:,:]
            y = y[0,:,:]
        noise_batch_size = z.shape[0]
        acc = []
        for i in range(noise_batch_size):
            res = sess.run(self.accuracy,feed_dict={self.xx:x,self.y_:y,self.z:z[[i],:]})
            acc.append(res)
        return np.array(acc)

    def Restore(self,sess:tf.Session, file_name):
        variables = tf.get_default_graph().get_collection('variables')
        reader = tf.train.NewCheckpointReader(file_name)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted([(var.name, var.name.split(':')[0]) for var in variables if var.name.split(':')[0] in saved_shapes])
        restore_vars = []
        name2var = dict(zip(map(lambda x: x.name.split(':')[0], variables), variables))
        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name in var_names:
                curr_var = name2var[saved_var_name]
                var_shape = curr_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(curr_var)
        saver = tf.train.Saver(restore_vars)
        sess.run(tf.global_variables_initializer())
        if os.path.isdir(file_name):
            saver.restore(sess, tf.train.latest_checkpoint(file_name))
        else:
            saver.restore(sess, file_name)

    def Normalize(self,w1, b1, w2, b2, w3, b3, w4, b4):
        fact = np.sqrt(np.sum(np.square(w1), (1, 2, 3)) + np.square(b1))
        fact = fact / (5 * 5 * 1)
        w1 = w1 / fact[:,np.newaxis,np.newaxis,np.newaxis,:]
        b1 = b1 / fact
        w2 = w2 * fact[:, np.newaxis, np.newaxis, :, np.newaxis]

        fact = np.sqrt(np.sum(np.square(w2), (1, 2, 3)) + np.square(b2))
        fact = fact / (5 * 5 * 32)
        w2 = w2 / fact[:,np.newaxis,np.newaxis,np.newaxis,:]
        b2 = b2 / fact
        w3 = w3 * fact[:, np.newaxis, np.newaxis, :, np.newaxis]

        fact = np.sqrt(np.sum(np.square(w3), (1, 2, 3)) + np.square(b3))
        fact = fact / (7 * 7 * 16)
        w3 = w3 / fact[:, np.newaxis, np.newaxis, np.newaxis, :]
        b3 = b3 / fact
        w4 = w4 * fact[:, :, np.newaxis]

        fact = np.sum(b4, (-1), keepdims=True)
        b4 = b4 - fact

        return w1, b1, w2, b2, w3, b3, w4, b4
