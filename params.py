class GeneralParameters():
    def __init__(self):
        # input data properties
        self.image_height = 28
        self.image_width = 28
        self.number_of_channels = 1
        self.number_of_categories = 10

        # seed for randomizing numpy and TensorFlow
        self.seed = 8734

class ClassifierHyperParameters():
    """
    hyperparameters for classification target network
    """
    def __init__(self):
        self.initialization_std = 0.1
        self.bias_initialization = 0.1

        # layer1 : convolutional
        self.layer1_filter_size = 5
        self.layer1_size = 32
        self.layer1_pool_size = 2

        # layer2 : convolutional
        self.layer2_filter_size = 5
        self.layer2_size = 16
        self.layer2_pool_size = 2

        # layer3 : fully connected
        self.layer3_size = 8

        # layer4 : fully connected (but has no params)


        self.fix_gauge = True

        # optimization params
        self.learning_rate = 1e-3
        self.learning_rate_rate = 0.9999
        self.batch_size = 256

class GeneratorHyperParameters():
    """
    hyperparameters for hypernetwork (=generator)
    """
    def __init__(self):
        self.leaky_relu_coeff = 0.05  # this is alpha, in max{x,alpha*x}

        self.with_batchnorm = True # should use batch normalization?
        self.batchnorm_decay = 0.98 # exponential decay constant for batch normalization

        self.input_noise_size = 300  # dimension of noise vector z
        self.input_noise_bound = 1  # z will be sampled from a uniform distribution [-input_noise_bound,input_noise_bound]
        self.e_layer_sizes = [300,300] # sizes of hidden layers for extractor
        self.code1_size = 15 # size of input code to weight generator 1
        self.code2_size = 15 # size of input code to weight generator 2
        self.code3_size = 15 # size of input code to weight generator 3
        self.code4_size = 15 # size of input code to weight generator 4
        self.w1_layer_sizes = [40,40] # sizes of hidden layers for weight generator 1
        self.w2_layer_sizes = [100,100] # sizes of hidden layers for weight generator 2
        self.w3_layer_sizes = [100,100] # sizes of hidden layers for weight generator 3
        self.w4_layer_sizes = [60,60] # sizes of hidden layers for weight generator 4

        self.fix_gauge = True

        self.zero_fixer = 1e-8 # add this constant to the argument of sqrt, log, etc. so that the argument is never zero

        self.initialization_std = 1e-1 # used for weights and biases
        self.noise_batch_size = 32 # batch size for noise, during training
        self.images_batch_size = 32  # batch size for images, during training
        self.noise_batch_size_for_validation = 200  # batch size for noise for validation
        self.learning_rate = 3e-4  # initial learning rate
        self.learning_rate_rate = 0.99998  # decay rate of learning rate - decay happens once every training step
        self.lamBda = 1e3  # 3e6 # initial lambda value (=coefficient of accuracy component in total loss)
        self.lambda_rate = 1.0  # growth rate of lambda - positive value mean anealing! rate is per training step
