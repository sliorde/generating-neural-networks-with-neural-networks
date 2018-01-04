import numpy as np
import tensorflow as tf

def GetWeightVariable(shape, name, scale=1.0):
    return tf.get_variable(shape=shape, initializer=tf.variance_scaling_initializer(scale=scale),name=name)

def GetBiasVariable(shape, name, scale=1.0):
    # return tf.get_variable(shape=shape, initializer=tf.variance_scaling_initializer(scale=scale),name=name)
    return tf.get_variable(shape=shape, initializer=tf.constant_initializer(0.0),trainable=False,name=name)

def MultiLayerPerceptron(input,widths,with_batch_norm=False,where_to_batch_norm=None,train_batch_norm=True,activation_on_last_layer=False,is_training=None,scale=1.0,batchnorm_decay=0.98,activation=tf.nn.relu):
    """
    Creates a multilayer perceptron (=MLP)
    :param input: input layer to perceptron
    :param widths: a list with the widths (=number of neurons) in each layer
    :param with_batch_norm: should perfrom batch normalization?
    :param where_to_batch_norm: (optional) a list with boolean values, one value for each layer. This determines whether to perform batch normalization in this layer.
    :param train_batch_norm: (optional) whether to train the batch normalization's offsets and scales, or just use constant values
    :param activation_on_last_layer: if boolean - whether to add activation on the last layer of the MLP. Otherwise, can be an actual activation function for the last layer
    :param is_training: tf boolean variable which determines whether currently the graph is in training phase. This determines which batch normalization value to use.
    :param scale: the variance of the initial values of the variables will be proportional to this
    :param batchnorm_decay:  exponential decay constant for batch normalization
    :param activation:
    :return: three lists: 1) layers: a list of tuples (w,b) where each tuple is the weights and biases for a layer. 2) layer_outputs: a list of tensors, each tensor is the output of a layer. 3) batch_norm_params: a list of tuples (means,variances,offsets,scales), the former two are tensors, the latter two are variables
    """
    inds = np.nonzero(widths)[0]
    widths = [widths[ind] for ind in inds]
    layers = []
    layer_outputs = [input,]
    widths = [int(input.shape[-1])]+widths
    if with_batch_norm and is_training is not None:
        if where_to_batch_norm is None:
            where_to_batch_norm = [False] + [True]*(len(widths)-1)
            if activation_on_last_layer is False:
                where_to_batch_norm[-1] = False
        else:
            where_to_batch_norm = [where_to_batch_norm[ind] for ind in inds]
        batch_norm_params = []
    else:
        batch_norm_params = None
        where_to_batch_norm = [False]*len(widths)
    for i in range(1,len(widths)):
        w = GetWeightVariable([widths[i - 1], widths[i]], 'layer{:d}_weights'.format(i), scale)
        b = GetBiasVariable([widths[i]], 'layer{:d}_biases'.format(i), scale)
        layers.append((w,b))
        pre_activations = tf.add(tf.tensordot(layer_outputs[i - 1], w, [[-1], [-2]]), b, 'layer{:d}_pre_activations'.format(i))
        if i<(len(widths)-1):
            layer_output = activation(pre_activations, 'layer{:d}_activations'.format(i))
        else:
            if activation_on_last_layer is False:
                layer_output = pre_activations
            elif activation_on_last_layer is True:
                layer_output = activation(pre_activations,'layer{:d}_activations'.format(i))
            else:
                layer_output = activation_on_last_layer(pre_activations, 'layer{:d}_activations'.format(i))
        if where_to_batch_norm[i]:
            batch_means,batch_variances = tf.nn.moments(layer_output,list(np.arange(0,len(layer_output.shape)-1)),keep_dims=False)
            offsets = tf.Variable(tf.zeros_like(batch_means),trainable=train_batch_norm,name='layer{:d}_offset'.format(i))
            scales = tf.Variable(tf.ones_like(batch_variances),trainable=train_batch_norm,name='layer{:d}_scale'.format(i))
            ema = tf.train.ExponentialMovingAverage(decay=batchnorm_decay,name='layer{:d}_EMA_for_batchnorm'.format(i))
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


def OptimizerReset(optimizer, graph=None, name=None):
    """
    reset all internal variables (=slots) of optimizer. It is important to do this  when doing a manual sharp change
    :param name:
    :return:
    """
    if graph is None:
        graph = tf.get_default_graph()
    slots = [optimizer.get_slot(var, name) for name in optimizer.get_slot_names() for var in graph.get_collection('variables')]
    slots = [slot for slot in slots if slot is not None]
    if isinstance(optimizer, tf.train.AdamOptimizer):
        slots.extend(optimizer._get_beta_accumulators())
    return tf.variables_initializer(slots, name=name)

def ExpandDims(tensor:tf.Tensor,axis,name=None):
    """
    perform multiple tf.expand_dims at once
    :param tensor:
    :param axis:
    :param name:
    :return:
    """
    for i in np.sort(axis):
        tensor = tf.expand_dims(tensor,i)
    tensor = tf.identity(tensor,name)
    return tensor

