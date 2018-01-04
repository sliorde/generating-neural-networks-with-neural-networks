import sys
sys.path.append('../')

from layers import *
from tensorflow.contrib import slim
from params import GeneralParameters,ClassifierHyperParameters

general_params = GeneralParameters()
classifier_hparams = ClassifierHyperParameters()


class MnfMnist(object):
    def __init__(self, N, input_shape, flows_q=2, flows_r=2, use_z=True,  activation=tf.nn.relu, logging=False,
                 nb_classes=general_params.number_of_categories, learn_p=False, layer_dims=(classifier_hparams.layer1_size, classifier_hparams.layer2_size, classifier_hparams.layer3_size), flow_dim_h=50, thres_var=1, prior_var_w=1.,
                 prior_var_b=1.):
        self.layer_dims = layer_dims
        self.activation = activation
        self.N = N
        self.input_shape = input_shape
        self.flows_q = flows_q
        self.flows_r = flows_r
        self.use_z = use_z

        self.logging = logging
        self.nb_classes = nb_classes
        self.flow_dim_h = flow_dim_h
        self.thres_var = thres_var
        self.learn_p = learn_p
        self.prior_var_w = prior_var_w
        self.prior_var_b = prior_var_b

        self.noise_size = classifier_hparams.layer1_size+classifier_hparams.layer1_filter_size*classifier_hparams.layer1_filter_size*general_params.number_of_channels*classifier_hparams.layer1_size+classifier_hparams.layer1_size+classifier_hparams.layer2_size+classifier_hparams.layer2_filter_size*classifier_hparams.layer2_filter_size*classifier_hparams.layer1_size*classifier_hparams.layer2_size+classifier_hparams.layer2_size+classifier_hparams.layer2_size * int((general_params.image_height/classifier_hparams.layer1_pool_size/classifier_hparams.layer2_pool_size)*(general_params.image_width/classifier_hparams.layer1_pool_size/classifier_hparams.layer2_pool_size))+classifier_hparams.layer2_size * classifier_hparams.layer3_size * int((general_params.image_height/classifier_hparams.layer1_pool_size/classifier_hparams.layer2_pool_size)*(general_params.image_width/classifier_hparams.layer1_pool_size/classifier_hparams.layer2_pool_size))+classifier_hparams.layer3_size+classifier_hparams.layer3_size+classifier_hparams.layer3_size*general_params.number_of_categories+general_params.number_of_categories

        self.z = tf.placeholder(tf.float32,[None,self.noise_size],'input_noise')

        self.opts = 'fq{}_fr{}_usez{}'.format(self.flows_q, self.flows_r, self.use_z)
        self.built = False

    def build_mnf_mnist(self, x, sample=True):
        if not self.built:
            self.layers = []
            epsZ0, epsW0, epsB0, epsZ1, epsW1, epsB1, epsZ2, epsW2, epsB2, epsZ3, epsW3, epsB3 = self.z_to_eps(self.z)
        with tf.variable_scope(self.opts):
            if not self.built:
                layer1 = Conv2DMNF(self.layer_dims[0], classifier_hparams.layer1_filter_size, classifier_hparams.layer1_filter_size, N=self.N, input_shape=self.input_shape, border_mode='SAME',
                                   flows_q=self.flows_q, flows_r=self.flows_r, logging=self.logging, use_z=self.use_z,
                                   learn_p=self.learn_p, prior_var=self.prior_var_w, prior_var_b=self.prior_var_b,
                                   thres_var=self.thres_var, flow_dim_h=self.flow_dim_h, epsZ=epsZ0,epsW=epsW0,epsB=epsB0)
                self.layers.append(layer1)
            else:
                layer1 = self.layers[0]
            h1 = self.activation(tf.nn.max_pool(layer1(x, sample=sample), [1, classifier_hparams.layer1_pool_size, classifier_hparams.layer1_pool_size, 1], [1, classifier_hparams.layer1_pool_size, classifier_hparams.layer1_pool_size, 1], 'SAME'))

            if not self.built:
                shape = [None] + [s.value for s in h1.get_shape()[1:]]
                layer2 = Conv2DMNF(self.layer_dims[1], classifier_hparams.layer2_filter_size, classifier_hparams.layer2_filter_size, N=self.N, input_shape=shape, border_mode='SAME',
                                   flows_q=self.flows_q, flows_r=self.flows_r, use_z=self.use_z, logging=self.logging,
                                   learn_p=self.learn_p, flow_dim_h=self.flow_dim_h, thres_var=self.thres_var,
                                   prior_var=self.prior_var_w, prior_var_b=self.prior_var_b, epsZ=epsZ1,epsW=epsW1,epsB=epsB1)
                self.layers.append(layer2)
            else:
                layer2 = self.layers[1]
            h2 = slim.flatten(self.activation(tf.nn.max_pool(layer2(h1, sample=sample), [1, classifier_hparams.layer2_pool_size, classifier_hparams.layer2_pool_size, 1], [1, classifier_hparams.layer2_pool_size, classifier_hparams.layer2_pool_size, 1], 'SAME')))

            if not self.built:
                fcinp_dim = h2.get_shape()[1].value
                layer3 = DenseMNF(self.layer_dims[2], N=self.N, input_dim=fcinp_dim, flows_q=self.flows_q,
                                  flows_r=self.flows_r, use_z=self.use_z, logging=self.logging, learn_p=self.learn_p,
                                  prior_var=self.prior_var_w, prior_var_b=self.prior_var_b, flow_dim_h=self.flow_dim_h,
                                  thres_var=self.thres_var, epsZ=epsZ2,epsW=epsW2,epsB=epsB2)
                self.layers.append(layer3)
            else:
                layer3 = self.layers[2]
            h3 = self.activation(layer3(h2, sample=sample))

            if not self.built:
                fcinp_dim = h3.get_shape()[1].value
                layerout = DenseMNF(self.nb_classes, N=self.N, input_dim=fcinp_dim, flows_q=self.flows_q,
                                    flows_r=self.flows_r, use_z=self.use_z, logging=self.logging, learn_p=self.learn_p,
                                    prior_var=self.prior_var_w, prior_var_b=self.prior_var_b, flow_dim_h=self.flow_dim_h,
                                    thres_var=self.thres_var, epsZ=epsZ3,epsW=epsW3,epsB=epsB3)
                self.layers.append(layerout)
            else:
                layerout = self.layers[3]

        if not self.built:
            self.built = True
        return layerout(h3, sample=sample)

    def predict(self, x, sample=True):
        return self.build_mnf_mnist(x, sample=sample)

    def get_reg(self):
        reg = 0.
        for j, layer in enumerate(self.layers):
            with tf.name_scope('kl_layer{}'.format(j + 1)):
                regi = layer.get_reg()
                tf.summary.scalar('kl_layer{}'.format(j + 1), regi)
            reg += regi

        return reg


    def z_to_eps(self,z):
        if isinstance(z,np.ndarray):
            reshape = np.reshape
        else:
            reshape = tf.reshape


        shape = [classifier_hparams.layer1_filter_size,classifier_hparams.layer1_filter_size,general_params.number_of_channels,classifier_hparams.layer1_size]
        start = 0
        stop = start+shape[-1]
        epsZ0 = z[:,start:stop]

        start = stop
        stop = start + np.prod(shape)
        epsW0 = reshape(z[:, start:stop],[-1]+shape)

        start = stop
        stop = start + shape[-1]
        epsB0 = z[:, start:stop]

        shape = [classifier_hparams.layer2_filter_size,classifier_hparams.layer2_filter_size,classifier_hparams.layer1_size,classifier_hparams.layer2_size]
        start = stop
        stop = start + shape[-1]
        epsZ1 = z[:, start:stop]

        start = stop
        stop = start + np.prod(shape)
        epsW1 = reshape(z[:, start:stop],[-1]+shape)

        start = stop
        stop = start + shape[-1]
        epsB1 = z[:, start:stop]

        shape = [int(general_params.image_height/classifier_hparams.layer1_pool_size/classifier_hparams.layer2_pool_size)*int(general_params.image_width/classifier_hparams.layer1_pool_size/classifier_hparams.layer2_pool_size)*classifier_hparams.layer2_size ,classifier_hparams.layer3_size]
        start = stop
        stop = start + shape[0]
        epsZ2 = z[:, start:stop]

        start = stop
        stop = start + np.prod(shape)
        epsW2 = reshape(z[:, start:stop],[-1]+shape)

        start = stop
        stop = start + shape[-1]
        epsB2 = z[:, start:stop]

        shape = [classifier_hparams.layer3_size, general_params.number_of_categories]
        start = stop
        stop = start + shape[0]
        epsZ3 = z[:, start:stop]

        start = stop
        stop = start + np.prod(shape)
        epsW3 = reshape(z[:, start:stop],[-1]+shape)

        start = stop
        stop = start + shape[-1]
        epsB3 = z[:, start:stop]

        return epsZ0,epsW0,epsB0,epsZ1,epsW1,epsB1,epsZ2,epsW2,epsB2,epsZ3,epsW3,epsB3
