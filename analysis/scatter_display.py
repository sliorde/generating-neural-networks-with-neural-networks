import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import pickle
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale as Scale
import sys

sys.path.append('../')
sys.path.append('../MNF')
from train import GetImages

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rc('font', size=8)
plt.rc('axes', titlesize=8)
plt.rc('axes', labelsize=8)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('legend', fontsize=8)
plt.rc('figure', titlesize=8)

np.random.seed(1212)

noise_batch_size = 3000

acc = None

# exactly one of these two sets of three lines need to be commented - depending in whether we want to use HyperNetwork or MnfNetwork

# from hypernetwork import HyperNetwork
# hnet = HyperNetwork()
# file_name = '../checkpoints/checkpoint-13000'

from mnf_network import MnfNetwork
hnet = MnfNetwork()
file_name = '../MNF/results/models/mnf_lenet_mnist_fq2_fr2_usezTrue_thres0.5/model/model'

z = hnet.SampleInput(noise_batch_size)
sess = tf.InteractiveSession()
hnet.Restore(sess,file_name)
w1,b1, w2,b2, w3,b3, w4,b4 = hnet.GenerateWeights(sess,z)
all_weights = np.concatenate([np.reshape(a,(noise_batch_size,-1)) for a in [w1,b1,w2,b2,w3,b3,w4,b4]],1)

def ShowLayer(w, b=None, acc=None, filt=None, do_scale=False):
    n = 6
    fig = plt.figure(figsize=(4, 4),facecolor='white')
    gs = gridspec.GridSpec(n, n)
    gs.update(wspace=0, hspace=0)

    w = np.reshape(w,[noise_batch_size,-1,w.shape[-1]])
    if b is not None:
        b = np.reshape(b, [noise_batch_size,-1, b.shape[-1]])



    if filt is None:
        filt = np.arange(0,w.shape[-1],dtype=np.int)
    if b is None:
        pre_transform = np.reshape(w[:, :, filt],[noise_batch_size,-1])
    else:
        pre_transform = np.concatenate((np.reshape(w[:, :, filt],[noise_batch_size,-1]),np.reshape(b[:, :, filt],[noise_batch_size,-1])), 1)
    # Scale(pre_transform, copy=False)
    transformed_weights = PCA(n_components=n).fit_transform(pre_transform)

    if acc is None:
        acc = np.zeros(noise_batch_size)
        cm = LinearSegmentedColormap.from_list(name='my_cm', colors=[(63/255, 72/255, 204/255), (63/255, 72/255, 204/255)], N=2)
        with_colorbar = False
    else:
        cm = LinearSegmentedColormap.from_list(name='my_cm', colors=[(0, 1, 0), (0, 0, 1)], N=255)
        with_colorbar = True

    for i in range(n):
        for j in range(n):
            if i<j:
                continue
            ax = plt.subplot(gs[i,j])
            if i == j:
                plt.hist(transformed_weights[:,i],30,color='skyblue',ec='blue',linewidth=0.4)
                ax.get_yaxis().set_ticks([])

            else:
                sc = ax.scatter(transformed_weights[:, j], transformed_weights[:, i], s=0.4,c=acc,linewidths=0)
                sc.set_cmap(cm)
                if j>0:
                    ax.get_yaxis().set_ticks([])
            if i < (n - 1):
                ax.get_xaxis().set_ticks([])
            ax.axis('tight')
            if i==(n-1):
                a = np.min(transformed_weights[:, j])
                b = np.max(transformed_weights[:, j])
                ax.get_xaxis().set_ticks([np.round(a+0.17*(b-a),2),np.round(b-0.17*(b-a),2)])
            if ((j==0)  and (i>j)):
                a = np.min(transformed_weights[:, i])
                b = np.max(transformed_weights[:, i])
                ax.get_yaxis().set_ticks([np.round(a+0.1*(b-a),2),np.round(b-0.1*(b-a),2)])
            ax.tick_params(axis='both', which='major', labelsize=6)


            if with_colorbar:
                if i == 1 and j == 0:
                    cbax = fig.add_axes([0.88, 0.38, 0.063, 0.60])
                    cb = plt.colorbar(cax=cbax,mappable=sc,orientation="vertical",ticks=[np.min(acc),np.max(acc)])
                    cbax.yaxis.set_ticks_position('left')
                    cbax.tick_params(axis='both', which='major', labelsize=6)


    plt.tight_layout(0,0,0)

    return fig


fig = ShowLayer(w1,b1,filt=np.random.randint(0,32),do_scale=True,acc=acc)

fig = ShowLayer(w2,b2,filt=np.random.randint(0,16),do_scale=True,acc=acc)

fig = ShowLayer(w3, b3, filt=np.random.randint(0, 8), do_scale=True, acc=acc)

fig = ShowLayer(w4, b4, filt=np.random.randint(0, 10), do_scale=True, acc=acc)


fig = ShowLayer(w1,b1,do_scale=True,acc=acc)

fig = ShowLayer(w2,b2,do_scale=True,acc=acc)

fig = ShowLayer(w3,b3,do_scale=True,acc=acc)

fig = ShowLayer(w4,b4,do_scale=True,acc=acc)


fig = ShowLayer(all_weights,do_scale=True,acc=acc)

plt.show()