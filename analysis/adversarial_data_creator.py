import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from scipy.stats import mode
import pickle
import sys

sys.path.append('../')
sys.path.append('../MNF/')

from train import GetImages


from params import GeneralParameters
general_params = GeneralParameters()

ensemble_size = 100
test_size = 10000
perturbations = np.arange(0,60,2)

# exactly one of these two sets of three lines need to be commented - depending in whether we want to use HyperNetwork or MnfNetwork
from hypernetwork import HyperNetwork
hnet = HyperNetwork()
file_name = '../parameters/checkpoint-18000'

# from mnf_network import MnfNetwork
# hnet = MnfNetwork()
# file_name = '../MNF/results/models/mnf_lenet_mnist_fq2_fr2_usezTrue_thres0.5/model/model'

sess = tf.InteractiveSession()
hnet.Restore(sess,file_name)
gradients = []
for i in range(general_params.number_of_categories):
    gradients.append(tf.gradients(hnet.probabilities[0,0,i],hnet.x))

min_pert_single1 = np.tile(-1.0, (test_size))
min_pert_ensemble1 = np.tile(-1.0, (test_size))

min_pert_single2 = np.tile(-1.0, (test_size))
min_pert_ensemble2 = np.tile(-1.0, (test_size))

x,y = GetImages('test')
y = np.nonzero(y[0, :, :])[1]

i = 0
im = 0
while (i < test_size) and (im<x.shape[1]):
    print(i)
    z = hnet.SampleInput(1)

    x_original = x[:,[im]]
    y_original = y[im]

    # make sure classifier isn't wrong on this image
    y_pred = hnet.Predict(sess,x_original,z)[0][0].astype(int)[0]
    if y_pred != y_original:
        im = im+1
        continue

    #choose the target (i.e a number 0-9)
    y_new = np.mod(y_original+np.random.randint(1,general_params.number_of_categories),general_params.number_of_categories).astype(int)

    # find gradient
    grad = sess.run(gradients[y_new], feed_dict={hnet.z: z, hnet.x: x_original})

    # create adversarial examples with all perturbation sizes
    correction = (1 / 255) * np.sign(grad[0])
    x_new = x_original + np.reshape(perturbations,(1,len(perturbations),1,1,1))*correction
    # x_new[x_new > 1] = 1
    # x_new[x_new < 0] = 0

    # prediction for the adversarial
    y_new_pred = hnet.Predict(sess,x_new,z)[0][0].astype(int)

    # find smallest perturbation size required
    try:
        min_pert_single1[i] = perturbations[np.nonzero(y_new_pred != y_original)[0][1]]
    except IndexError:
        min_pert_single1[i] = np.Inf
    try:
        min_pert_single2[i] = perturbations[np.nonzero(y_new_pred == y_new)[0][1]]
    except IndexError:
        min_pert_single2[i] = np.Inf


    # test adversarial on ensemble (by voting`)
    z = hnet.SampleInput(ensemble_size)
    y_new_preds_ensemble = np.zeros((ensemble_size, len(perturbations)))
    for j in range(ensemble_size):
        y_new_preds_ensemble[j, :] = hnet.Predict(sess,x_new,z[[j],:])[0][0].astype(int)
    y_new_pred_ensemble = np.squeeze(mode(y_new_preds_ensemble, 0)[0])

    # find smallest perturbation size required
    try:
        min_pert_ensemble1[i] = perturbations[np.nonzero(y_new_pred_ensemble != y_original)[0][1]]
    except IndexError:
        min_pert_ensemble1[i] = np.Inf
    try:
        min_pert_ensemble2[i] = perturbations[np.nonzero(y_new_pred_ensemble == y_new)[0][1]]
    except IndexError:
        min_pert_ensemble2[i] = np.Inf


    i += 1
    im += 1

actual_test_size = i

min_pert_single1 = min_pert_single1[:actual_test_size]
min_pert_ensemble1 = min_pert_ensemble1[:actual_test_size]
min_pert_single2 = min_pert_single2[:actual_test_size]
min_pert_ensemble2 = min_pert_ensemble2[:actual_test_size]

d = perturbations[1]-perturbations[0]
cdf_single1 = np.cumsum(np.histogram(min_pert_single1[np.isfinite(min_pert_single1)],np.concatenate((perturbations[[0]]-d/2,perturbations+d/2)))[0])/actual_test_size
cdf_ensemble1 = np.cumsum(np.histogram(min_pert_ensemble1[np.isfinite(min_pert_ensemble1)],np.concatenate((perturbations[[0]]-d/2,perturbations+d/2)))[0])/actual_test_size
cdf_single2 = np.cumsum(np.histogram(min_pert_single2[np.isfinite(min_pert_single2)],np.concatenate((perturbations[[0]]-d/2,perturbations+d/2)))[0])/actual_test_size
cdf_ensemble2 = np.cumsum(np.histogram(min_pert_ensemble2[np.isfinite(min_pert_ensemble2)],np.concatenate((perturbations[[0]]-d/2,perturbations+d/2)))[0])/actual_test_size

data = {'cdf_single1':cdf_single1,'cdf_ensemble1':cdf_ensemble1,'cdf_single2':cdf_single2,'cdf_ensemble2':cdf_ensemble2,'perturbations':perturbations}
with open("data_adversarial.pickle","wb") as f:
    pickle.dump(data, f)