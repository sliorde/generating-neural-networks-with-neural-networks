import numpy as np
import tensorflow as tf
import os
from scipy.special import gamma as gamma_function

from utils import *
from params import  GeneralParameters,ClassifierHyperParameters,GeneratorHyperParameters

class HyperNetwork():
    def __init__(self, general_params:GeneralParameters=None,classifier_hparams:ClassifierHyperParameters=None,generator_hparams:GeneratorHyperParameters=None, use_generator=True,graph=None):
        """

        :param general_params:
        :param classifier_hparams: classifier target network hyperparameters
        :param generator_hparams: hypernetwork (=generator) hyperparameters
        :param use_generator: whether to use the hypernetwork (=generaor) for generating weights, or just use weight as trainable variables (which means performing conventional training)
        :param graph: a tensorflow graph, where the hypernetwork will be built
        """
        if general_params is None:
            general_params = GeneralParameters()
        self.general_params=general_params

        if classifier_hparams is None:
            classifier_hparams = ClassifierHyperParameters()
        self.classifier_hparams=classifier_hparams

        if generator_hparams is None:
            generator_hparams = GeneratorHyperParameters()
        self.generator_hparams = generator_hparams

        self.use_generator = use_generator

        if graph is None:
            graph = tf.get_default_graph()
        self.graph = graph

        self.Build()

    def Build(self):
        """
        create the graph
        """
        with self.graph.as_default():
            with tf.variable_scope('optimization'):
                step_counter = tf.Variable(0, trainable=False,name='step_counter')  # counts how many training step were  performed

            if self.use_generator:
                with tf.variable_scope('generator_input'):
                    z = tf.placeholder(tf.float32, shape=[None, self.generator_hparams.input_noise_size], name='input_noise') # primary input to generator
                    is_training = tf.Variable(False,trainable=False,name='is_training') # this variable can be fed from outside, but its default value is False. influences only batchnorm

                # this will be the function that will be used to build the extractor and weight generators
                mlp_builder = lambda input,widths: MultiLayerPerceptron(input, widths, with_batch_norm=self.generator_hparams.with_batchnorm,scale=np.square(self.generator_hparams.initialization_std),batchnorm_decay=self.generator_hparams.batchnorm_decay,activation=self.Activation,is_training=tf.where(step_counter>0,is_training,tf.constant(True)))

                with tf.variable_scope('extractor'):
                    output_size = self.generator_hparams.code1_size * self.classifier_hparams.layer1_size + self.generator_hparams.code2_size * self.classifier_hparams.layer2_size + self.generator_hparams.code3_size*self.classifier_hparams.layer3_size + self.generator_hparams.code4_size
                    e_layers, e_layer_outputs, e_batch_norm_params = mlp_builder(z, self.generator_hparams.e_layer_sizes+[output_size])

                next_ind = 0
                with tf.variable_scope('weight_generator1'):
                    start_ind = next_ind
                    next_ind = start_ind +  self.generator_hparams.code1_size * self.classifier_hparams.layer1_size
                    codes1 = tf.reshape(e_layer_outputs[-1][:,start_ind:next_ind],[-1,self.classifier_hparams.layer1_size,self.generator_hparams.code1_size],'codes')
                    output_size = self.classifier_hparams.layer1_filter_size*self.classifier_hparams.layer1_filter_size*self.general_params.number_of_channels+1
                    w1_layers, w1_layer_outputs, w1_batch_norm_params = mlp_builder(codes1, self.generator_hparams.w1_layer_sizes+[output_size])

                with tf.variable_scope('weight_generator2'):
                    start_ind = next_ind
                    next_ind = start_ind +  self.generator_hparams.code2_size * self.classifier_hparams.layer2_size
                    codes2 = tf.reshape(e_layer_outputs[-1][:,start_ind:next_ind],[-1,self.classifier_hparams.layer2_size,self.generator_hparams.code2_size],'codes')
                    output_size = self.classifier_hparams.layer2_filter_size*self.classifier_hparams.layer2_filter_size*self.classifier_hparams.layer1_size+1
                    w2_layers, w2_layer_outputs, w2_batch_norm_params = mlp_builder(codes2, self.generator_hparams.w2_layer_sizes+[output_size])

                with tf.variable_scope('weight_generator3'):
                    start_ind = next_ind
                    next_ind = start_ind +  self.generator_hparams.code3_size * self.classifier_hparams.layer3_size
                    codes3 = tf.reshape(e_layer_outputs[-1][:,start_ind:next_ind],[-1,self.classifier_hparams.layer3_size,self.generator_hparams.code3_size],'codes')
                    output_size = int(((self.general_params.image_height / (self.classifier_hparams.layer1_pool_size * self.classifier_hparams.layer2_pool_size)) * (self.general_params.image_width / (self.classifier_hparams.layer1_pool_size * self.classifier_hparams.layer2_pool_size)) * self.classifier_hparams.layer2_size) + 1)
                    w3_layers, w3_layer_outputs, w3_batch_norm_params = mlp_builder(codes3, self.generator_hparams.w3_layer_sizes+[output_size])

                with tf.variable_scope('weight_generator4'):
                    start_ind = next_ind
                    next_ind = start_ind +  self.generator_hparams.code4_size
                    codes4 = tf.identity(e_layer_outputs[-1][:,start_ind:next_ind],'codes')
                    output_size = self.general_params.number_of_categories*(self.classifier_hparams.layer3_size + 1)
                    w4_layers, w4_layer_outputs, w4_batch_norm_params = mlp_builder(codes4, self.generator_hparams.w4_layer_sizes+[output_size])

            with tf.variable_scope('classifier_input'):
                x = tf.placeholder(tf.float32, [None, None, self.general_params.image_height, self.general_params.image_width, self.general_params.number_of_channels],name='input_images')
                y = tf.placeholder(tf.float32, [None, None, self.general_params.number_of_categories], name='labels')

            with tf.variable_scope('weights'):
                if self.use_generator:
                    w1_not_gauged = w1_layer_outputs[-1][:,:,0:(-1)]
                    b1_not_gauged = w1_layer_outputs[-1][:,:,(-1)]
                    w1_not_gauged = tf.transpose(tf.reshape(w1_not_gauged,[-1,self.classifier_hparams.layer1_size,self.classifier_hparams.layer1_filter_size,self.classifier_hparams.layer1_filter_size,self.general_params.number_of_channels]),[0,2,3,4,1],name='w1_not_gauged')
                    b1_not_gauged = tf.identity(b1_not_gauged,name='b1_not_gauged')

                    w2_not_gauged = w2_layer_outputs[-1][:,:,0:(-1)]
                    b2_not_gauged = w2_layer_outputs[-1][:,:,(-1)]
                    w2_not_gauged = tf.transpose(tf.reshape(w2_not_gauged,[-1,self.classifier_hparams.layer2_size,self.classifier_hparams.layer2_filter_size,self.classifier_hparams.layer2_filter_size,self.classifier_hparams.layer1_size]),[0,2,3,4,1],name='w2_not_gauged')
                    b2_not_gauged = tf.identity(b2_not_gauged,name='b2_not_gauged')

                    w3_not_gauged = w3_layer_outputs[-1][:,:,0:(-1)]
                    b3_not_gauged = w3_layer_outputs[-1][:,:,(-1)]
                    w3_not_gauged = tf.transpose(tf.reshape(w3_not_gauged,[-1,self.classifier_hparams.layer3_size,int(self.general_params.image_height / (self.classifier_hparams.layer1_pool_size * self.classifier_hparams.layer2_pool_size)),int(self.general_params.image_width / (self.classifier_hparams.layer1_pool_size * self.classifier_hparams.layer2_pool_size)),self.classifier_hparams.layer2_size]),[0,2,3,4,1],name='w3_not_gauged')
                    b3_not_gauged = tf.identity(b3_not_gauged,name='b3_not_gauged')

                    w4_not_gauged = w4_layer_outputs[-1][:, 0:(self.classifier_hparams.layer3_size*self.general_params.number_of_categories)]
                    b4_not_gauged = w4_layer_outputs[-1][:, (self.classifier_hparams.layer3_size*self.general_params.number_of_categories):]
                    w4_not_gauged = tf.reshape(w4_not_gauged,[-1,self.classifier_hparams.layer3_size,self.general_params.number_of_categories],name='w4_not_gauged')
                    b4_not_gauged = tf.identity(b4_not_gauged, name='b4_not_gauged')
                else:
                    get_weight_variable = lambda shape,name: tf.Variable(tf.truncated_normal(shape,stddev=self.classifier_hparams.initialization_std),name=name)
                    get_bias_variable = lambda shape,name: tf.Variable(tf.constant(self.classifier_hparams.bias_initialization,shape=shape), name=name)
                    w1_not_gauged = get_weight_variable([1,self.classifier_hparams.layer1_filter_size,self.classifier_hparams.layer1_filter_size,self.general_params.number_of_channels,self.classifier_hparams.layer1_size],name='w1_not_gauged')
                    b1_not_gauged = get_bias_variable([1,self.classifier_hparams.layer1_size], name='b1_not_gauged')
                    w2_not_gauged = get_weight_variable([1,self.classifier_hparams.layer2_filter_size,self.classifier_hparams.layer2_filter_size,self.classifier_hparams.layer1_size,self.classifier_hparams.layer2_size],name='w2_not_gauged')
                    b2_not_gauged = get_bias_variable([1,self.classifier_hparams.layer2_size], name='b2_not_gauged')
                    w3_not_gauged = get_weight_variable([1,int((self.general_params.image_height / (self.classifier_hparams.layer1_pool_size * self.classifier_hparams.layer2_pool_size))),int((self.general_params.image_width / (self.classifier_hparams.layer1_pool_size * self.classifier_hparams.layer2_pool_size))),self.classifier_hparams.layer2_size,self.classifier_hparams.layer3_size],name='w3_not_gauged')
                    b3_not_gauged = get_bias_variable([1,self.classifier_hparams.layer3_size], name='b3_not_gauged')
                    w4_not_gauged = get_weight_variable([1,self.classifier_hparams.layer3_size,self.general_params.number_of_categories],name='w4_not_gauged')
                    b4_not_gauged = get_bias_variable([1,self.general_params.number_of_categories], name='b4_not_gauged')

            # now we perform a gauge transformation to bring the weights into the required gauge
            with tf.variable_scope('gauge_fixing'):
                if (self.use_generator and self.generator_hparams.fix_gauge) or ((not self.use_generator) and self.classifier_hparams.fix_gauge):
                    required_scale = self.classifier_hparams.layer1_filter_size*self.classifier_hparams.layer1_filter_size*self.general_params.number_of_channels+1
                    scale_factor = tf.sqrt((tf.reduce_sum(tf.square(w1_not_gauged), [1, 2, 3]) + tf.square(b1_not_gauged))/required_scale+self.generator_hparams.zero_fixer)
                    w1 = w1_not_gauged / (ExpandDims(scale_factor, [1, 2, 3]) + self.generator_hparams.zero_fixer)
                    b1 = b1_not_gauged / (scale_factor + self.generator_hparams.zero_fixer)
                    w2 = w2_not_gauged * ExpandDims(scale_factor, [1, 2, 4])
                    w1 = tf.identity(w1,'w1')
                    b1 = tf.identity(b1, 'b1')

                    required_scale = self.classifier_hparams.layer2_filter_size*self.classifier_hparams.layer2_filter_size*self.classifier_hparams.layer1_size+1
                    scale_factor = tf.sqrt((tf.reduce_sum(tf.square(w2), [1, 2, 3]) + tf.square(b2_not_gauged))/required_scale+self.generator_hparams.zero_fixer)
                    w2 = w2 / (ExpandDims(scale_factor, [1, 2, 3]) + self.generator_hparams.zero_fixer)
                    b2 = b2_not_gauged / (scale_factor + self.generator_hparams.zero_fixer)
                    w3 = w3_not_gauged * ExpandDims(scale_factor, [1, 2, 4])
                    w2 = tf.identity(w2, 'w2')
                    b2 = tf.identity(b2, 'b2')

                    required_scale = (self.general_params.image_height / (self.classifier_hparams.layer1_pool_size * self.classifier_hparams.layer2_pool_size)) * (self.general_params.image_width / (self.classifier_hparams.layer1_pool_size * self.classifier_hparams.layer2_pool_size)) * self.classifier_hparams.layer2_size + 1
                    scale_factor = tf.sqrt((tf.reduce_sum(tf.square(w3), [1, 2, 3]) + tf.square(b3_not_gauged))/required_scale+self.generator_hparams.zero_fixer)
                    w3 = w3 / (ExpandDims(scale_factor, [1, 2, 3]) + self.generator_hparams.zero_fixer)
                    b3 = b3_not_gauged / (scale_factor + self.generator_hparams.zero_fixer)
                    w4 = w4_not_gauged * ExpandDims(scale_factor, [2])
                    w3 = tf.identity(w3, 'w3')
                    b3 = tf.identity(b3, 'b3')

                    required_softmax_bias = 0.0
                    softmax_bias_diff = tf.reduce_sum(b4_not_gauged, 1,keep_dims=True) - required_softmax_bias
                    b4 = b4_not_gauged - softmax_bias_diff
                    w4 = tf.identity(w4, 'w4')
                    b4 = tf.identity(b4, 'b4')
                else:
                    w1 = tf.identity(w1_not_gauged,'w1')
                    b1 = tf.identity(b1_not_gauged,'b1')
                    w2 = tf.identity(w2_not_gauged,'w2')
                    b2 = tf.identity(b2_not_gauged,'b2')
                    w3 = tf.identity(w3_not_gauged,'w3')
                    b3 = tf.identity(b3_not_gauged,'b3')
                    w4 = tf.identity(w4_not_gauged,'w4')
                    b4 = tf.identity(b4_not_gauged,'b4')


            with tf.variable_scope('classifier_network'):
                fn = lambda u: tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(u[0], u[1], padding='SAME', strides=[1, 1, 1, 1]) + u[2]),ksize=[1, self.classifier_hparams.layer1_pool_size, self.classifier_hparams.layer1_pool_size, 1],strides=[1, self.classifier_hparams.layer1_pool_size, self.classifier_hparams.layer1_pool_size, 1],padding='SAME')
                c_layer1_output = tf.map_fn(fn, elems=[x, w1, b1], dtype=tf.float32,name='layer1_output')

                fn = lambda u: tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(u[0], u[1], padding='SAME', strides=[1, 1, 1, 1]) + u[2]),ksize=[1, self.classifier_hparams.layer2_pool_size, self.classifier_hparams.layer2_pool_size, 1],strides=[1, self.classifier_hparams.layer2_pool_size, self.classifier_hparams.layer2_pool_size, 1],padding='SAME')
                c_layer2_output = tf.map_fn(fn, elems=[c_layer1_output, w2, b2], dtype=tf.float32,name='layer2_output')

                c_layer3_output = tf.nn.relu(tf.reduce_sum(tf.expand_dims(c_layer2_output, -1) * tf.expand_dims(w3,1), axis=(2, 3, 4)) + tf.expand_dims(b3,1),name='layer3_output')

                c_layer4_output = tf.identity(tf.reduce_sum(tf.expand_dims(c_layer3_output, -1) * tf.expand_dims(w4,1), axis=2) + tf.expand_dims(b4,1),name='layer4_output')

                probabilities = tf.nn.softmax(c_layer4_output,name='probabilities')
                predictions = tf.argmax(probabilities, axis=2, name='prediction')


            with tf.variable_scope('helpful_variables'):
                if self.use_generator:
                    noise_batch_size = tf.identity(tf.shape(z)[0],name='noise_batch_size')
                else:
                    noise_batch_size = tf.constant(1,name='noise_batch_size')
                image_batch_size = tf.identity(tf.shape(y)[1],'image_batch_size')

                correct_predictions = tf.equal(predictions, tf.argmax(y, axis=2), name='correct_prediction')
                accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), axis=1, name='accuracy')
                average_accuracy = tf.reduce_mean(accuracy, name='average_accuracy')

                flattened_network = tf.concat(axis=1, values=[tf.reshape(w1, [noise_batch_size, -1]),tf.reshape(b1, [noise_batch_size, -1]),tf.reshape(w2, [noise_batch_size, -1]),tf.reshape(b2, [noise_batch_size, -1]),tf.reshape(w3, [noise_batch_size, -1]),tf.reshape(b3, [noise_batch_size, -1]),tf.reshape(w4, [noise_batch_size, -1]),tf.reshape(b4, [noise_batch_size, -1])],name='flattened_network')

            with tf.variable_scope('loss'):
                accuracy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.reshape(y,shape=(-1,self.general_params.number_of_categories)), logits=tf.reshape(c_layer4_output,shape=(-1,self.general_params.number_of_categories))),name='accuracy_loss')

                if self.use_generator:
                    # entropy estimated using  Kozachenko-Leonenko estimator, with l1 distances
                    mutual_distances = tf.reduce_sum(tf.abs(tf.expand_dims(flattened_network, 0) - tf.expand_dims(flattened_network, 1)), 2,name='mutual_squared_distances') # all distances between weight vector samples
                    nearest_distances = tf.identity(-1*tf.nn.top_k(-1 * mutual_distances, k=2)[0][:, 1] ,name='nearest_distances') # distance to nearest neighboor for each weight vector sample
                    entropy_estimate = tf.identity(self.generator_hparams.input_noise_size * tf.reduce_mean(tf.log(nearest_distances + self.generator_hparams.zero_fixer)) + tf.digamma(tf.cast(noise_batch_size, tf.float32)), name='entropy_estimate')

                    diversity_loss = tf.identity( - 1 * entropy_estimate, name='diversity_loss')

                    lamBda = tf.Variable(self.generator_hparams.lamBda, dtype=tf.float32, trainable=False, name='lambda')
                    loss = tf.identity(lamBda*accuracy_loss + diversity_loss,name='loss')
                else:
                    loss = tf.identity(accuracy_loss,name='loss')

            with tf.variable_scope('optimization'):
                if self.use_generator:
                    learning_rate = self.generator_hparams.learning_rate
                    learning_rate_rate = self.generator_hparams.learning_rate_rate
                else:
                    learning_rate = self.classifier_hparams.learning_rate
                    learning_rate_rate = self.classifier_hparams.learning_rate_rate
                learning_rate = tf.Variable(learning_rate, dtype=tf.float32,trainable=False, name='learning_rate')
                learning_rate_rate = tf.Variable(learning_rate_rate, dtype=tf.float32,trainable=False, name='learning_rate_rate') # rate of change of learning rate (per 1 training step)
                update_learning_rate = tf.assign(learning_rate,learning_rate*learning_rate_rate,name='update_learning_rate') # op for decaying learning rate
                steps_before_train_step = [update_learning_rate]

                if self.use_generator:
                    lambda_rate = tf.Variable(self.generator_hparams.lambda_rate, dtype=tf.float32,trainable=False, name='lambda_rate') # rate of change of lambda (per 1 training step)
                    update_lambda = tf.assign(lamBda, lamBda * lambda_rate, name='update_lambda') # op for increasing lambda (=annealing)
                    steps_before_train_step.append(update_lambda)

                optimizer = tf.train.AdamOptimizer(learning_rate, name='optimizer_adam')
                with tf.control_dependencies(steps_before_train_step):
                    train_step = optimizer.minimize(loss,name='train_step')
                    with tf.control_dependencies([train_step]):
                        step_counter_update = tf.assign_add(step_counter, 1, name='step_counter_update')

                train_step = tf.group(*(steps_before_train_step+[train_step,step_counter_update]),name='update_and_train')
                reset_optimizer = OptimizerReset(optimizer,self.graph,'adam_resetter') # reset optimizer's internal variable. use this function when manually updating learning rate or lambda

            with tf.variable_scope('initializer'):
                initializer = tf.variables_initializer(self.graph.get_collection('variables'),name='initializer')

            with tf.variable_scope('saver'):
                saver = tf.train.Saver(max_to_keep=100)

        if self.use_generator:
            self.z = z
            self.is_training = is_training
            self.e_layers = e_layers
            self.e_layer_outputs = e_layer_outputs
            self.e_batch_norm_params = e_batch_norm_params

            self.codes1 = codes1
            self.w1_layers = w1_layers
            self.w1_layer_outputs = w1_layer_outputs
            self.w1_batch_norm_params = w1_batch_norm_params

            self.codes2 = codes2
            self.w2_layers = w2_layers
            self.w2_layer_outputs = w2_layer_outputs
            self.w2_batch_norm_params = w2_batch_norm_params

            self.codes3 = codes3
            self.w3_layers = w3_layers
            self.w3_layer_outputs = w3_layer_outputs
            self.w3_batch_norm_params = w3_batch_norm_params

            self.codes4 = codes4
            self.w4_layers = w4_layers
            self.w4_layer_outputs = w4_layer_outputs
            self.w4_batch_norm_params = w4_batch_norm_params

            self.mutual_distances = mutual_distances
            self.nearest_distances = nearest_distances
            self.entropy_estimate = entropy_estimate
            self.diversity_loss = diversity_loss

            self.lamBda = lamBda
            self.lambda_rate = lambda_rate
            self.update_lambda = update_lambda

        self.x = x
        self.y = y
        self.w1 = w1
        self.w1_not_gauged = w1_not_gauged
        self.b1 = b1
        self.b1_not_gauged = b1_not_gauged
        self.w2 = w2
        self.w2_not_gauged = w2_not_gauged
        self.b2 = b2
        self.b2_not_gauged = b2_not_gauged
        self.w3 = w3
        self.w3_not_gauged = w3_not_gauged
        self.b3 = b3
        self.b3_not_gauged = b3_not_gauged
        self.w4 = w4
        self.w4_not_gauged = w4_not_gauged
        self.b4 = b4
        self.b4_not_gauged = b4_not_gauged

        self.c_layer1_output = c_layer1_output
        self.c_layer2_output = c_layer2_output
        self.c_layer3_output = c_layer3_output
        self.c_layer4_output = c_layer4_output

        self.probabilities = probabilities
        self.prediction = predictions

        self.noise_batch_size = noise_batch_size
        self.image_batch_size = image_batch_size
        self.correct_prediction = correct_predictions
        self.accuracy = accuracy
        self.average_accuracy = average_accuracy
        self.flattened_network = flattened_network
        self.accuracy_loss = accuracy_loss

        self.loss = loss
        self.learning_rate = learning_rate
        self.learning_rate_rate = learning_rate_rate
        self.update_learning_rate = update_learning_rate
        self.optimizer = optimizer
        self.reset_optimizer = reset_optimizer
        self.train_step = train_step
        self.step_counter = step_counter
        self.step_counter_update = step_counter_update

        self.Initializer = initializer
        self.saver = saver


    def SampleInput(self,batch_size=None,task=None):
        """
        sample from z, the input distribution to the hypernetwork
        :param batch_size:
        :param task: (optional) 'train' or 'validation'. will be ignored if a batch size was specified
        :return: a sample of z
        """
        if batch_size is None:
            if task=='train':
                batch_size = self.generator_hparams.noise_batch_size
            elif task=='validation':
                batch_size = self.generator_hparams.noise_batch_size_for_validation
            else:
                batch_size=1
        return np.random.uniform(-1 * self.generator_hparams.input_noise_bound, self.generator_hparams.input_noise_bound, size=[batch_size, self.generator_hparams.input_noise_size])

    def Activation(self,input, name=None):
        output = tf.maximum(self.generator_hparams.leaky_relu_coeff * input, input, name)
        return output

    def GenerateWeights(self,sess:tf.Session,z=None,noise_batch_size=None):
        """
        return all weights of target classifier network
        :param sess:
        :param z:
        :param noise_batch_size:
        :return: w1,b1,w2,b2,w3,b3.w4,b4
        """
        if self.use_generator:
            if z is None:
                if noise_batch_size is None:
                    noise_batch_size = 1
                z = self.SampleInput(noise_batch_size)
            else:
                noise_batch_size = z.shape[0]
            return sess.run([self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4],feed_dict={self.z:z})
        else:
            return sess.run([self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4])

    def Predict(self,sess:tf.Session,x,z=None,noise_batch_size=None,step_size=None):
        """
        predict classes of input x
        :param sess:
        :param x:
        :param z:
        :param noise_batch_size:
        :return: prediction, probabilities
        """
        if self.use_generator:
            if z is None:
                if noise_batch_size is None:
                    noise_batch_size = 1
                z = self.SampleInput(noise_batch_size)
            return self.GetMetrics(sess, [self.prediction, self.probabilities], x=x, z=z, step_size=step_size)
        else:
            return self.GetMetrics(sess, [self.prediction, self.probabilities], x=x)

    def PredictWithForcedWeights(self,sess:tf.Session,x,w1,b1,w2,b2,w3,b3,w4,b4,step_size=None):
        """
        force the weights of the target classification network to have certain values, and use them for prediction
        :param sess:
        :param x:
        :param w1:
        :param b1:
        :param w2:
        :param b2:
        :param w3:
        :param b3:
        :param w4:
        :param b4:
        :return: prediction, probabilities
        """
        if w1.ndim == 4:
            w1 = np.expand_dims(w1, 0)
            b1 = np.expand_dims(b1, 0)
            w2 = np.expand_dims(w2, 0)
            b2 = np.expand_dims(b2, 0)
            w3 = np.expand_dims(w3, 0)
            b3 = np.expand_dims(b3, 0)
            w4 = np.expand_dims(w4, 0)
            b4 = np.expand_dims(b4, 0)
            noise_batch_size = 1
        else:
            noise_batch_size = w1.shape[0]
        if x.ndim == 4:
            x = np.expand_dims(x,0)
        if x.shape[0] == 1:
            x = np.tile(x,[noise_batch_size,1,1,1,1])

        if step_size is None:
            return sess.run([self.prediction,self.probabilities],feed_dict={self.x:x,self.w1:w1,self.b1:b1,self.w2:w2,self.b2:b2,self.w3:w3,self.b3:b3,self.w4:w4,self.b4:b4})
        else:
            preds = []
            probs = []
            i = 0
            while i < noise_batch_size:
                start = i
                end = i+step_size
                end = np.minimum(end,noise_batch_size)
                feed_dict = {self.w1:w1[start:end],self.b1:b1[start:end],self.w2:w2[start:end],self.b2:b2[start:end],self.w3:w3[start:end],self.b3:b3[start:end],self.w4:w4[start:end],self.b4:b4[start:end],self.x:x[:(end-start)]}
                pred,prob = sess.run([self.prediction,self.probabilities],feed_dict)
                preds.append(pred)
                probs.append(prob)
                i = end
            preds = np.concatenate(preds,0)
            probs = np.concatenate(probs, 0)

            return preds,probs


    def GetAccuracyWithForcedWeights(self,sess:tf.Session,x,y,w1,b1,w2,b2,w3,b3,w4,b4,step_size=None):
        """
        force the weights of the target classification network to have certain values, and use them for accuracy calculation
        :param sess:
        :param x:
        :param w1:
        :param b1:
        :param w2:
        :param b2:
        :param w3:
        :param b3:
        :param w4:
        :param b4:
        :return: prediction, probabilities
        """
        if not self.use_generator:
            step_size = 1
        if w1.ndim == 4:
            w1 = np.expand_dims(w1, 0)
            b1 = np.expand_dims(b1, 0)
            w2 = np.expand_dims(w2, 0)
            b2 = np.expand_dims(b2, 0)
            w3 = np.expand_dims(w3, 0)
            b3 = np.expand_dims(b3, 0)
            w4 = np.expand_dims(w4, 0)
            b4 = np.expand_dims(b4, 0)
            noise_batch_size = 1
        else:
            noise_batch_size = w1.shape[0]
        if x.ndim == 4:
            x = np.expand_dims(x,0)
            y = np.expand_dims(y, 0)
        if x.shape[0] == 1:
            x = np.tile(x,[noise_batch_size,1,1,1,1])
            y = np.tile(y, [noise_batch_size, 1,1])

        if step_size is None:
            accs = sess.run(self.accuracy,feed_dict={self.x:x,self.y:y,self.w1:w1,self.b1:b1,self.w2:w2,self.b2:b2,self.w3:w3,self.b3:b3,self.w4:w4,self.b4:b4})
        else:
            accs = []
            i = 0
            while i < noise_batch_size:
                start = i
                end = i+step_size
                end = np.minimum(end,noise_batch_size)
                feed_dict = {self.x:x[:(end-start)],self.y:y[:(end-start)],self.w1:w1[start:end],self.b1:b1[start:end],self.w2:w2[start:end],self.b2:b2[start:end],self.w3:w3[start:end],self.b3:b3[start:end],self.w4:w4[start:end],self.b4:b4[start:end]}
                acc = sess.run(self.accuracy,feed_dict)
                accs.append(acc)
                i = end
            accs = np.concatenate(accs,0)

        return accs

    def GetAccuracy(self,sess:tf.Session,x,y,z=None,step_size=5):
        return self.GetMetrics(sess, self.accuracy, x, y, z, step_size=step_size)

    def NumberOfParameters(self):
        """

        :return: number of trainable parameters in hypernetwork
        """
        tot = np.sum([np.prod(var.get_shape()) for var in self.graph.get_collection('trainable_variables')])
        return tot

    def NumberOfWeights(self):
        """

        :return: number of weights (=parameters) in target classification network
        """
        num_of_params_layer1 = (self.classifier_hparams.layer1_size * (self.classifier_hparams.layer1_filter_size * self.classifier_hparams.layer1_filter_size * self.general_params.number_of_channels+ 1))
        num_of_params_layer2 = (self.classifier_hparams.layer2_size * (self.classifier_hparams.layer2_filter_size * self.classifier_hparams.layer2_filter_size * self.classifier_hparams.layer1_size + 1))
        num_of_params_layer3 = (self.classifier_hparams.layer3_size * ((self.general_params.image_height / self.classifier_hparams.layer1_pool_size / self.classifier_hparams.layer2_pool_size) * (self.general_params.image_width / self.classifier_hparams.layer1_pool_size / self.classifier_hparams.layer2_pool_size) * self.classifier_hparams.layer2_size + 1))
        num_of_params_layer4 = (self.general_params.number_of_categories * (self.classifier_hparams.layer3_size + 1))
        num_of_params = num_of_params_layer1 + num_of_params_layer2 + num_of_params_layer3 + num_of_params_layer4
        return num_of_params

    def TrainStep(self,sess:tf.Session,x,y,z=None,noise_batch_size=None):
        """

        :param sess:
        :param x:
        :param y:
        :param z:
        :param noise_batch_size:
        :return: how many training steps were performed so far (in total)
        """
        if self.use_generator:
            if z is None:
                if noise_batch_size is None:
                    noise_batch_size = self.generator_hparams.noise_batch_size
                z = self.SampleInput(noise_batch_size)
            else:
                noise_batch_size = z.shape[0]
            if x.ndim == 4:
                x = np.expand_dims(x, 0)
                y = np.expand_dims(y,0)
            if x.shape[0] == 1:
                x = np.tile(x, [noise_batch_size, 1, 1, 1, 1])
                y = np.tile(y, [noise_batch_size, 1, 1])
            sess.run(self.train_step, feed_dict={self.z: z, self.x: x, self.y: y, self.is_training: True})
        else:
            if x.ndim == 4:
                x = np.expand_dims(x, 0)
                y = np.expand_dims(y,0)
            sess.run(self.train_step,feed_dict={self.x:x,self.y:y})

        return self.GetStepCounter(sess)

    def GetLossFromComponents(self,sess,accuracy_loss,diversity_loss):
        """
        calculate total loss from accuracy loss and diversity loss
        :param sess:
        :param accuracy_loss:
        :param diversity_loss:
        :return:
        """
        return sess.run(self.loss,feed_dict={self.accuracy_loss:accuracy_loss,self.diversity_loss:diversity_loss})

    def GetMetrics(self,sess:tf.Session,metrics,x=None,y=None,z=None,is_training=False,step_size=None):
        """
        fetch values from graph
        :param sess:
        :param metrics: variables\tensors in hypernet (list or scalar)
        :param x:
        :param y:
        :param z:
        :param is_training:
        :param step_size: if not None, will use GetMetricsLoop() instead
        :return:
        """

        if (step_size is not None) and self.use_generator:
            return self.GetMetricsLoop(sess,metrics,x,y,z,step_size,is_training)
        else:
            feed_dict = {}
            if self.use_generator:
                if z is None:
                    noise_batch_size = 1
                else:
                    noise_batch_size = z.shape[0]
                    feed_dict[self.z] = z
                feed_dict[self.is_training] = is_training
            else:
                noise_batch_size = 1
            if x is not None:
                if x.ndim == 4:
                    x = np.expand_dims(x,0)
                if x.shape[0] == 1:
                    x = np.tile(x,[noise_batch_size,1,1,1,1])
                feed_dict[self.x] = x
            if y is not None:
                if y.ndim == 2:
                    y = np.expand_dims(y,0)
                if y.shape[0] == 1:
                    y = np.tile(y,[noise_batch_size,1,1])
                feed_dict[self.y] = y
            return sess.run(metrics,feed_dict)

    def GetMetricsLoop(self,sess:tf.Session,metrics,x=None,y=None,z=None,step_size=1,is_training=False):
        """
        use this to fetch values from graph in the case where batch size (images and noise) is too big for GPU. This will use a loop for the fetching
        :param sess:
        :param metrics: variables\tensors in hypernet (list or scalar)
        :param x:
        :param y:
        :param z:
        :param step_size: how many noise samples per loop iteration
        :param is_training:
        :return:
        """
        feed_dict = {}
        if self.use_generator:
            feed_dict[self.is_training] = is_training
        if x is not None:
            if x.ndim == 4:
                x = np.expand_dims(x, 0)
            if x.shape[0] == 1:
                x = np.tile(x, [step_size, 1, 1, 1, 1])
        if y is not None:
            if y.ndim == 2:
                y = np.expand_dims(y, 0)
            if y.shape[0] == 1:
                y = np.tile(y, [step_size, 1, 1])
        if isinstance(metrics, (list, tuple)):
            out = [[] for i in range(len(metrics))]
        else:
            out = []
        i = 0
        while i < z.shape[0]:
            start = i
            end = i+step_size
            end = np.minimum(end,z.shape[0])
            feed_dict[self.z] = z[start:end,:]
            if x is not None:
                feed_dict[self.x] = x[:(end-start), :,:,:,:]
            if y is not None:
                feed_dict[self.y] = y[:(end-start),:,:]
            results = sess.run(metrics,feed_dict)
            if isinstance(metrics, (list, tuple)):
                for j,res in enumerate(results):
                    if np.isscalar(res):
                        res = np.tile(res,end-start)
                    out[j].append(res)
            else:
                out.append(results)
            i = end
        if isinstance(metrics, (list, tuple)):
            for j,_ in enumerate(out):
                out[j] = np.concatenate(out[j],0)
        else:
            out = np.concatenate(out,0)
        return out

    def GetStepCounter(self,sess:tf.Session):
        """
        return how many training steps were performed so far
        :param sess:
        :return:
        """
        return sess.run(self.step_counter)

    def SaveToCheckpoint(self,sess:tf.Session,filename):
        self.saver.save(sess,filename,global_step=self.GetStepCounter(sess))

    def Restore(self,sess:tf.Session, file_name):
        """
        this function restores variables from checkpoint, and makes sure to initialize any variables which do not appear in the checkpoint
        :param sess:
        :param file_name:
        :return: how many training steps were performed so far
        """
        variables = self.graph.get_collection('variables')
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
        self.Initialize(sess)
        if os.path.isdir(file_name):
            saver.restore(sess, tf.train.latest_checkpoint(file_name))
        else:
            saver.restore(sess, file_name)

        return self.GetStepCounter(sess)

    def Initialize(self,sess:tf.Session):
        """
        initialize variables
        :param sess:
        :return: how many training steps were performed so far (should be zero)
        """
        sess.run(self.Initializer)

        return self.GetStepCounter(sess)

    def UpdateVariableFromFile(self, sess:tf.Session, var, filename):
        """
        update a variable to the value given in the file. After update, the file is deleted.
        :param sess:
        :param var: a tf.Variable()
        :param filename:
        :return: (new_value,old_value)
        """
        try:
            with open(filename) as fl:
                lines = fl.readlines()
            new_value = float(lines[0].strip())
            current_value = sess.run(var)
            sess.run(self.reset_optimizer)
            sess.run(var.assign(new_value))
            os.remove(filename)
            return new_value,current_value
        except FileNotFoundError:
            return None,None

    def UpdateLearningRateFromFile(self, sess:tf.Session, filename):
        """
        update the learning rate to the value given in the file. After update, the file is deleted.
        :param sess:
        :param filename:
        :return: (new_value,old_value)
        """
        return self.UpdateVariableFromFile(sess, self.learning_rate, filename)

    def UpdateLearningRateRateFromFile(self, sess:tf.Session, filename):
        """
        update the learning rate decay rate to the value given in the file. After update, the file is deleted.
        :param sess:
        :param filename:
        :return: (new_value,old_value)
        """
        return self.UpdateVariableFromFile(sess, self.learning_rate_rate, filename)

    def UpdateLambdaFromFile(self, sess:tf.Session, filename):
        """
        update lambda to the value given in the file. After update, the file is deleted.
        :param sess:
        :param filename:
        :return: (new_value,old_value)
        """
        if self.use_generator:
            return self.UpdateVariableFromFile(sess, self.lamBda, filename)
        else:
            return None,None

    def UpdateLambdaRateFromFile(self, sess:tf.Session, filename):
        """
        update the anealing rate to the value given in the file. After update, the file is deleted.
        :param sess:
        :param filename:
        :return: (new_value,old_value)
        """
        if self.use_generator:
            return self.UpdateVariableFromFile(sess, self.lambda_rate, filename)
        else:
            return None, None