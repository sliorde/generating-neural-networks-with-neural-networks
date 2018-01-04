import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['patch.linewidth'] = 0.6
plt.rc('font', size=8)
plt.rc('axes', titlesize=8)
plt.rc('axes', labelsize=8)
plt.rc('axes', linewidth=0.6)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('legend', fontsize=8)
plt.rc('figure', titlesize=8)


seed1 = 69319
seed2 = 32031
np.random.seed(seed1)
tf.set_random_seed(seed2)

evaluate_batch_size = 400

borders = 9

from toy_model import *

evaluation_set = np.linspace(-1.0,1.0,evaluate_batch_size)[:,np.newaxis]

sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver.restore(sess,'./toy')


y_out,al,dl,l,ff,lyrs = sess.run([y,accuracy_loss,diversity_loss,loss,f,layers],feed_dict={z:evaluation_set})
dists = np.sqrt(np.sum(np.square(y_out[1:, :] - y_out[:(-1), :]), 1))
dists = np.array([dists[0]] + [np.minimum(dists[i], dists[i + 1]) for i in range(len(dists) - 1)] + [dists[-1]])


fig = plt.figure(figsize=(2, 2),facecolor='white')
ax = plt.axes()
x1,x2 = np.meshgrid(np.linspace(-borders,borders,700),np.linspace(-borders,borders,700))
x = np.stack((x1.flatten(),x2.flatten()),1)
fff = sess.run(f,feed_dict={y:x})
fff = fff.reshape((700,700))
im = ax.imshow(fff,extent=[-borders,borders,-borders,borders],cmap=plt.get_cmap('Oranges'),origin='lower')
sc = ax.scatter(y_out[:,0],y_out[:,1],s=0.8,c=evaluation_set,cmap=plt.get_cmap('winter'),linewidths=0)
ax.set_xlim(-borders, borders)
ax.set_ylim(-borders, borders)
ax.get_yaxis().set_ticks([])
ax.get_xaxis().set_ticks([])
plt.tight_layout(0,0,0)


fig = plt.figure(figsize=(2, 1),facecolor='white')
ax = plt.axes()
sc = ax.scatter(evaluation_set,ff,s=3,c=evaluation_set,cmap=plt.get_cmap('winter'),linewidths=0)
ax.set_xlim(np.min(evaluation_set), np.max(evaluation_set))
ax.set_ylim(0, np.max(ff)*1.01)
ax.get_yaxis().set_ticks([])
ax.tick_params(axis='both', which='major', labelsize=6)
plt.tight_layout(0,0,0)

fig = plt.figure(figsize=(2, 1),facecolor='white')
ax = plt.axes()
sc = ax.scatter(evaluation_set,dists,s=3,c=evaluation_set,cmap=plt.get_cmap('winter'),linewidths=0)
ax.set_xlim(np.min(evaluation_set), np.max(evaluation_set))
ax.set_ylim(0, np.max(dists)*1.01)
ax.get_yaxis().set_ticks([])
ax.tick_params(axis='both', which='major', labelsize=6)
plt.tight_layout(0,0,0)



plt.show()