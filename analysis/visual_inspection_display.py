import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import sys

sys.path.append('../')
sys.path.append('../MNF')


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIX`ral'

noise_batch_size = 25

# exactly one of these two sets of three lines need to be commented - depending in whether we want to use HyperNetwork or MnfNetwork

# from hypernetwork import HyperNetwork
# hnet = HyperNetwork()
# file_name = '../checkpoints/checkpoint-13000'

from mnf_network import MnfNetwork
hnet = MnfNetwork()
file_name = '../MNF/results/models/mnf_lenet_mnist_fq2_fr2_usezTrue_thres0.5/model/model'

z = hnet.SampleInput(noise_batch_size)
sess = tf.InteractiveSession(graph=hnet.graph)
hnet.Restore(sess,file_name)
w1,b1, w2,b2, w3,b3, w4,b4 = hnet.GenerateWeights(sess,z)

def ShowFilters(w,filt=0,channel=0):
    n = 5
    fig = plt.figure(figsize=(2.2, 2.2), facecolor='white')
    gs = gridspec.GridSpec(n, n)
    gs.update(wspace=0, hspace=0)
    for i in range(n):
        for j in range(n):
            ax = plt.subplot(gs[i,j])
            im = ax.imshow(w[n*i+j,:,:,channel,filt], cmap=plt.get_cmap('Blues_r'), interpolation='nearest')
            ax.get_yaxis().set_ticks([])
            ax.get_xaxis().set_ticks([])
            ax.axis('tight')
    plt.tight_layout(0,0,0)


fig = ShowFilters(w1,filt=np.random.randint(0,32))
ShowFilters(w2,filt=np.random.randint(0,16),channel=np.random.randint(0,32))
ShowFilters(w3,filt=np.random.randint(0,8),channel=np.random.randint(0,16))
ShowFilters(np.expand_dims(np.expand_dims(w4,-1),-1))
plt.show()