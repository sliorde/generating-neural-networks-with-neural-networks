import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from scipy import stats
import pickle
import sys

sys.path.append('../')
sys.path.append('../MNF/')
from train import GetImages

from params import GeneralParameters
general_params = GeneralParameters()

noise_batch_size = 40
number_of_paths = 10

from hypernetwork import HyperNetwork

# exactly one of these two sets of lines need to be commented - depending in whether we want to use HyperNetwork or MnfNetwork

# hnet = HyperNetwork()
# file_name = 'checkpoint-13000'
# noise_size =hnet.generator_hparams.input_noise_size

from mnf_network import MnfNetwork
hnet = MnfNetwork()
file_name = '../MNF/results/models/mnf_lenet_mnist_fq2_fr2_usezTrue_thres0.5/model/model'
noise_size = hnet.noise_size



net = HyperNetwork(use_generator=False)

x,y = GetImages('test')
labels = np.nonzero(y[0, :, :])[1] # convert one-hot to regular representation

with tf.Session() as sess:
    hnet.Restore(sess,file_name)

    acc1 = []
    acc2 = []
    zs = []


    for i in range(number_of_paths):

        z = hnet.SampleInput(2)
        w1,b1, w2,b2, w3,b3, w4,b4= hnet.GenerateWeights(sess,z)

        zs.append(z)

        interp = lambda q: (q[[-1]]-q[[0]])*np.reshape(np.linspace(0,1,noise_batch_size),[-1]+[1]*(q.ndim-1))+q[[0]]
        z = interp(z)
        w1 = interp(w1)
        b1 = interp(b1)
        w2 = interp(w2)
        b2 = interp(b2)
        w3 = interp(w3)
        b3 = interp(b3)
        w4 = interp(w4)
        b4 = interp(b4)

        acc1.append(hnet.GetAccuracy(sess,x,y,z))
        acc2.append(net.GetAccuracyWithForcedWeights(sess,x,y,w1,b1, w2,b2, w3,b3, w4,b4))


data = {'accuracy_direct':acc2,'accuracy_interp':acc1,'z':zs}
with open("path.pickle","wb") as f:
    pickle.dump(data, f)

