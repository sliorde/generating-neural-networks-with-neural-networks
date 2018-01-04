import numpy as np
import tensorflow as tf
from scipy.special import gamma as gamma_function

def MultiLayerPerceptron(input,widths,without_biases=False,with_batch_norm=False,where_to_batch_norm=None,train_batch_norm=True,activation=tf.nn.relu,activation_on_last_layer=False,is_training=None,scale=1.0):
    inds = np.nonzero(widths)[0]
    widths = [widths[ind] for ind in inds]
    layers = []
    layer_outputs = [input,]
    widths = [int(input.shape[-1])]+widths
    if with_batch_norm and is_training is not None:
        if where_to_batch_norm is None:
            where_to_batch_norm = [False] + [True]*(len(widths)-1)
            if not activation_on_last_layer:
                where_to_batch_norm[-1] = False
        else:
            where_to_batch_norm = [where_to_batch_norm[ind] for ind in inds]
        batch_norm_params = []
    else:
        batch_norm_params = None
        where_to_batch_norm = [False]*len(widths)
    for i in range(1,len(widths)):
        w = tf.get_variable(shape=[widths[i - 1], widths[i]], initializer=tf.variance_scaling_initializer(scale), name='layer{:d}_weights'.format(i))
        if without_biases:
            b = tf.zeros(shape=[widths[i]], name='layer{:d}_biases'.format(i))
        else:
            b = tf.get_variable(shape=[widths[i]], initializer=tf.variance_scaling_initializer(scale),name='layer{:d}_biases'.format(i))
        layers.append((w,b))
        pre_activations = tf.add(tf.tensordot(layer_outputs[i - 1], w, [[-1], [-2]]), b, 'layer{:d}_pre_activations'.format(i))
        if (not activation_on_last_layer) and (i==(len(widths)-1)):
            layer_output = pre_activations
        else:
            layer_output = activation(pre_activations,'layer{:d}_activations'.format(i))
        if where_to_batch_norm[i]:
            batch_means,batch_variances = tf.nn.moments(layer_output,0,keep_dims=False)
            offsets = tf.Variable(tf.zeros_like(batch_means),trainable=train_batch_norm,name='layer{:d}_offset'.format(i))
            scales = tf.Variable(tf.ones_like(batch_variances),trainable=train_batch_norm,name='layer{:d}_scale'.format(i))
            ema = tf.train.ExponentialMovingAverage(decay=0.98,name='layer{:d}_EMA_for_batchnorm'.format(i))
            def ApplyEmaUpdate():
                ema_apply_op = ema.apply([batch_means, batch_variances])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_means), tf.identity(batch_variances)
            means,variances = tf.cond(is_training, lambda: ApplyEmaUpdate(),lambda: (ema.average(batch_means), ema.average(batch_variances)))
            layer_output = tf.nn.batch_normalization(layer_output,means,variances,offsets,scales,1e-7,'layer{:d}_batch_normalized_activations'.format(i))
            batch_norm_params.append((means,variances,offsets,scales))
        layer_outputs.append(layer_output)
    layer_outputs = layer_outputs[1:]
    return layers,layer_outputs,batch_norm_params

def RandomInput(size):
    return np.random.uniform(-1,1,[size,1])

train_batch_size = 5
lamda = 2.2
gamma = 0.05 # this is the coefficient of the regularization term
learning_rate = 0.001
learning_rate_rate = 0.99996
momentum = 0.0
with_batch_norm = False
activation = tf.nn.tanh
scale = 6

# generate a random mixture of Gaussians
sz = np.random.randint(10,11)
mu = np.random.uniform(-5,5,[sz,2])
sigma = np.random.uniform(1.0,1.1,[sz])
p = np.random.uniform(1,3,[sz])
p = p/np.sum(p)

z = tf.placeholder(tf.float32,[None,1],'input_noise')

layers,layer_outputs,batch_norm_params = MultiLayerPerceptron(z,[30,10,10,2],with_batch_norm=with_batch_norm,activation=activation,scale=scale)

y = tf.identity(layer_outputs[-1],'output')


f = tf.reduce_sum(tf.stack([p[i]*2*np.pi*np.square(sigma[i])*tf.exp(-1*tf.reduce_sum(tf.square(y-np.expand_dims(mu[i],0)),1)/(2*np.square(sigma[i]))) for i in range(len(p))],1),1)
accuracy_loss = -1*tf.reduce_mean(f)

# calculate entropy estimation
mutual_squared_distances = tf.reduce_sum(tf.square(tf.expand_dims(y, 0) - tf.expand_dims(y, 1)), 2,name='mutual_squared_distances')
nearest_distances = tf.sqrt(-1*tf.nn.top_k(-1 * mutual_squared_distances, k=2)[0][:, 1]+1e-7,name='nearest_distances')
entropy_estimate = tf.identity(tf.reduce_mean(tf.log(nearest_distances+1e-7)) + tf.digamma(tf.cast(train_batch_size, tf.float32)) + np.euler_gamma + (0.5) * np.log(np.pi) - np.log(gamma_function(1 + 1 / 2)), name='entropy_estimate')
diversity_loss = -1*entropy_estimate

# this is the l2 regularization
gauge_fixing = tf.reduce_mean(tf.reduce_sum(tf.square(y-0.0),1),name='gauge_fixing')

loss = accuracy_loss+lamda*diversity_loss + gamma*gauge_fixing

learning_rate_ = tf.Variable(learning_rate, dtype=tf.float32,trainable=False, name='learning_rate')
update_learning_rate = tf.assign(learning_rate_,learning_rate_*learning_rate_rate,name="update_learning_rate")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_,name='optimizer')
train_step = optimizer.minimize(loss,name='train_step')
train_step = tf.group(update_learning_rate,train_step,name='update_and_train')



saver = tf.train.Saver()