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

ensemble_size = 200
number_of_ensembles = 20

# exactly one of these two sets of three lines needs to be commented - depending in whether we want to use HyperNetwork or MnfNetwork
from hypernetwork import HyperNetwork
hnet = HyperNetwork()
file_name = '../parameters/checkpoint-11000'

# from mnf_network import MnfNetwork
# hnet = MnfNetwork()
# file_name = '../MNF/results/models/mnf_lenet_mnist_fq2_fr2_usezTrue_thres0.5/model/model'

x,y = GetImages('test')
labels = np.nonzero(y[0, :, :])[1] # convert one-hot to regular representation

preds,probs = [],[]
with tf.Session() as sess:
    hnet.Restore(sess,file_name)

    for i in range(number_of_ensembles):
        print(i)
        z = hnet.SampleInput(ensemble_size)
        pred,prob = hnet.Predict(sess,x,z,step_size=5)
        preds.append(pred)
        probs.append(prob)

preds = np.stack(preds,0)
probs = np.stack(probs,0)

accs_per_z = np.mean(preds==np.reshape(labels,(1,1,-1)),2)  # accuracy for each z
total_acc = np.mean(accs_per_z)

pred1 = stats.mode(preds,1)[0]  # ensemble prediction using majority vote
acc1 = np.mean(pred1 == np.reshape(labels,(1,1,-1)),2) # accuracy of each ensemble
pred2 = np.argmax(np.mean(probs, axis=1), axis=2)  # ensemble prediction using maximum mean probabilities
acc2 = np.mean(pred2 == np.reshape(labels,(1,-1)),1) # accuracy of each ensemble

data = {'ensemble_sizes':ensemble_size,'number_of_ensembles':number_of_ensembles,'accs_per_z':accs_per_z,'total_acc':total_acc,'acc1':acc1,'acc2':acc2}
with open("accuracies.pickle","wb") as f:
    pickle.dump(data, f)