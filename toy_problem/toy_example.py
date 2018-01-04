import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

seed1 = 69319 # change this to generate new objective function
seed2 = 55 # change this to change inital weights of network
np.random.seed(seed1)
tf.set_random_seed(seed2)

evaluate_batch_size = 1000

borders = 9 # borders of image

from toy_model import *

evaluation_set = np.linspace(-1.0,1.0,evaluate_batch_size)[:,np.newaxis]

sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


fig, ax = plt.subplots(3,1)
x1,x2 = np.meshgrid(np.linspace(-borders,borders,700),np.linspace(-borders,borders,700))
x = np.stack((x1.flatten(),x2.flatten()),1)
ff = sess.run(f,feed_dict={y:x})
ff = ff.reshape((700,700))
im = ax[0].imshow(ff,extent=[-borders,borders,-borders,borders],cmap=plt.get_cmap('Oranges'),origin='lower',vmin=np.percentile(ff,0))
sc = ax[0].scatter(np.zeros(evaluate_batch_size),np.zeros(evaluate_batch_size),s=4,c=evaluation_set,cmap=plt.get_cmap('winter'),linewidths=0,animated=True)
sc2 = ax[1].scatter(evaluation_set,np.zeros(evaluate_batch_size),s=4,c=evaluation_set,cmap=plt.get_cmap('winter'),linewidths=0,animated=True)
sc3 = ax[2].scatter(evaluation_set,np.zeros(evaluate_batch_size),s=4,c=evaluation_set,cmap=plt.get_cmap('winter'),linewidths=0,animated=True)
y_max = [0.1]
dists_max = [0.0001]

ax[0].set_xlim(-borders, borders)
ax[0].set_ylim(-borders, borders)
ax[1].set_xlim(-1, 1)
ax[1].set_ylim(0, y_max[0])
ax[2].set_xlim(-1, 1)
ax[2].set_ylim(0, dists_max[0])

def init():
    return sc,sc2,sc3


def update(frame):
    print('----{:d}----'.format(frame))
    for i in range(20):
            sess.run(train_step, feed_dict={z: RandomInput(train_batch_size)})
    y_out,al,dl,l,ff= sess.run([y,accuracy_loss,diversity_loss,loss,f],feed_dict={z:evaluation_set})
    lyrs = sess.run(layers)
    sc.set_offsets(y_out)
    sc2.set_offsets(np.stack((np.squeeze(evaluation_set),ff),1))
    dists = np.sqrt(np.sum(np.square(y_out[1:,:]-y_out[:(-1),:]),1))
    dists =np.array([dists[0]]+[np.minimum(dists[i],dists[i+1]) for i in range(len(dists)-1)]+[dists[-1]])
    # sc.set_color(sc.get_cmap()(dists/1.0))
    dists_max[0] = np.maximum(dists_max[0], np.max(dists))
    sc3.set_offsets(np.stack((np.squeeze(evaluation_set), dists), 1))
    ax[2].set_ylim(0, dists_max[0])
    y_max[0] = np.maximum(y_max[0],np.max(ff))
    ax[1].set_ylim(0,y_max[0])
    print('{:.4f}   {:.4f}   {:.4f}'.format(al,dl,l))
    print('  ')
    if frame%500 == 0:
        saver.save(sess,'./toy')
    return sc,sc2,sc3

ami = FuncAnimation(fig,update,frames=np.arange(0,1000000),init_func=init,blit=True)
plt.show()
