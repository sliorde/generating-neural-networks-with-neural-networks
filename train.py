import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
from scipy import stats
import logging
import os

from params import GeneralParameters,ClassifierHyperParameters,GeneratorHyperParameters
from hypernetwork import HyperNetwork

# import ipdb; ipdb.set_trace()

CHECKPOINT_FILENAME = './checkpoints/checkpoint'
INITIALIZE_FROM_CHECKPOINT = False
CHECKPOINT_STEP = 0 # will return to this step number
CHECKPOINT_MESSAGE = None # a message to be written to the log file at initialization (optional)


# to update the learning_rate etc, create a text file and enter the new value (the file will be deleted after the update)
LEARNING_RATE_FILE_NAME = 'lr.txt'
LEARNING_RATE_RATE_FILE_NAME = 'lrr.txt'
LAMBDA_FILE_NAME = 'lambda.txt'
LAMBDA_RATE_FILE_NAME = 'lambda_rate.txt'

USE_GENERATOR = True # whether to train hypernetwork (=generator); otherwise will train just classifier

general_params = GeneralParameters()

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
np.random.seed(general_params.seed)
tf.set_random_seed(general_params.seed)

mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)

def InitializeLogger():
    if INITIALIZE_FROM_CHECKPOINT:
        log_file_mode = 'a'
    else:
        log_file_mode = 'w'
        if not os.path.exists(os.path.dirname(CHECKPOINT_FILENAME)):
            os.makedirs(os.path.dirname(CHECKPOINT_FILENAME))
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger()
    file_name = os.path.dirname(CHECKPOINT_FILENAME)+'/log.txt'
    file_handler = logging.FileHandler(file_name, mode=log_file_mode)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

def InitializeVariables(sess:tf.Session,hnet:HyperNetwork):
    """

    :param sess:
    :param hnet:
    :return: the current step counter
    """
    if INITIALIZE_FROM_CHECKPOINT:
        i = hnet.Restore(sess, CHECKPOINT_FILENAME+'-'+str(CHECKPOINT_STEP))
        logging.info("\n")
        logging.info("======INITIALIZED FROM CHECKPOINT======")
        logging.info("RETURNED TO STEP {:d}".format(i))
        if CHECKPOINT_MESSAGE is not None:
            logging.info("additional message:  " + CHECKPOINT_MESSAGE + "\n")
        logging.info("\n")
    else:
        i = hnet.Initialize(sess)
    return i

def GetImages(which_set='train', image_batch_size=None,noise_batch_size=1):
    if which_set=='train':
        which_set = mnist.train
    elif which_set=='validation':
        which_set = mnist.validation
    else:
        which_set = mnist.test

    if (image_batch_size is None) or (noise_batch_size is None):
        x = np.reshape(which_set.images, (1, -1, general_params.image_height, general_params.image_width, general_params.number_of_channels))
        y = np.reshape(which_set.labels, (1, -1, general_params.number_of_categories))
    else:
        if image_batch_size is None:
            image_batch_size = 1
        if noise_batch_size is None:
            noise_batch_size = 1
        batch = which_set.next_batch(image_batch_size*noise_batch_size)
        x = np.reshape(batch[0], (noise_batch_size, image_batch_size, general_params.image_height, general_params.image_width,general_params.number_of_channels))
        y = np.reshape(batch[1], (noise_batch_size, image_batch_size, general_params.number_of_categories))

    return x,y

def UpdateStuff(sess:tf.Session,hnet:HyperNetwork):
    """
    used to manually update the learning rate, learning rate rate (=decay), lambda, and lambda rate (=annealing). To perform an update, the user needs to create a text file with the new value, The file will be deleted once the update is done
    :param sess:
    :param hnet:
    :return:
    """
    new,old = hnet.UpdateLearningRateFromFile(sess, LEARNING_RATE_FILE_NAME)
    if new is not None:
        logging.info("--> Changed learning rate from {:.3e} to {:.3e} (factor of {:.1f})".format(old,new,max([old/ new, new/old])))
    new,old = hnet.UpdateLearningRateRateFromFile(sess, LEARNING_RATE_RATE_FILE_NAME)
    if new is not None:
        logging.info("--> Changed learning rate rate from {:.6e} to {:.6e} (factor of {:.5f})".format(old,new,max([old/ new, new/old])))
    new,old = hnet.UpdateLambdaFromFile(sess, LAMBDA_FILE_NAME)
    if new is not None:
        logging.info("--> Changed lambda from {:.3e} to {:.3e} (factor of {:.1f})".format(old,new,max([old/ new, new/old])))
    new,old = hnet.UpdateLambdaRateFromFile(sess, LAMBDA_RATE_FILE_NAME)
    if new is not None:
        logging.info("--> Changed lambda rate from {:.6e} to {:.6e} (factor of {:.5f})".format(old,new,max([old/ new, new/old])))

def TrainClassifier(hnet:HyperNetwork,max_steps=1e6):
    """
    train classifier, without using generator (=hypernetwork)
    :param hnet:
    :param max_steps:
    :return:
    """
    InitializeLogger()
    x_validation,y_validation = GetImages('validation')

    with tf.Session(graph=hnet.graph) as sess:
        i = InitializeVariables(sess,hnet)
        t = time.time()
        while i < max_steps:
            if i % 1000 == 0:
                accuracy, loss, learning_rate = hnet.GetMetrics(sess,[hnet.average_accuracy,hnet.loss,hnet.learning_rate],x_validation,y_validation)
                logging.info("\n")
                logging.info("step {:d}, test accuracy {:.4f},    loss {:.4f}".format(i, accuracy, loss))
                logging.info("learning rate {:.4f}".format(learning_rate))
                logging.info('elapsed: {:.2f} minutes'.format((time.time() - t) / 60))
                logging.info("\n")

                hnet.SaveToCheckpoint(sess,CHECKPOINT_FILENAME)

            x,y = GetImages('train',hnet.classifier_hparams.batch_size)
            if i % 100 == 0:
                accuracy, loss = hnet.GetMetrics(sess, [hnet.average_accuracy, hnet.loss], x, y)
                logging.info("step {:d}, training accuracy {:.4f},    loss {:.4f}".format(i, accuracy, loss))

                # check if there are new values for updating learning rate and lambda
                UpdateStuff(sess,hnet)

            i = hnet.TrainStep(sess,x,y)

def TrainGenerator(hnet:HyperNetwork,max_steps=1e6):
    """
    train hypernetwork (=generator)
    :param hnet:
    :param max_steps:
    :return:
    """
    InitializeLogger()

    x_validation, y_validation = GetImages('validation')
    labels = np.nonzero(y_validation[0, :, :])[1] # convert one-hot to regular representation

    with tf.Session(graph=hnet.graph) as sess:
        i = InitializeVariables(sess,hnet)
        while i<=max_steps:
            if (i % 1000 == 0):
                z = hnet.SampleInput(task='validation')
                accuracies, probs, preds, accuracy_loss = hnet.GetMetrics(sess,[hnet.accuracy, hnet.probabilities, hnet.prediction,hnet.accuracy_loss],x_validation,y_validation,z,step_size=5)
                entropy,diversity_loss, w1, b1, w2, b2, w3, b3, w4, b4, learning_rate,learning_rate_rate, lamBda, lambda_rate = hnet.GetMetrics(sess,[hnet.entropy_estimate, hnet.diversity_loss,hnet.w1, hnet.b1, hnet.w2, hnet.b2, hnet.w3, hnet.b3, hnet.w4, hnet.b4, hnet.learning_rate, hnet.learning_rate_rate, hnet.lamBda, hnet.lambda_rate],z=z)

                accuracy = np.mean(accuracies)
                accuracy_loss = np.mean(accuracy_loss)
                total_loss = hnet.GetLossFromComponents(sess,accuracy,diversity_loss)

                pred1 = np.squeeze(stats.mode(preds)[0]) # ensemble prediction using majority vote
                acc1 = np.sum(pred1 == labels) / len(labels)  # accuracy of ensemble
                pred2 = np.argmax(np.mean(probs, axis=0), axis=1) # ensemble prediction using maximum mean probabilities
                acc2 = np.sum(pred2 == labels) / len(labels) # accuracy of ensemble
                pred3 = np.argmax(np.max(probs,0),-1) # ensemble prediction using most confident individual
                acc3 = np.sum(pred3 == labels) / len(labels) # accuracy of ensemble

                cnts, bins = np.histogram(accuracies, 6)

                logging.info("\n\n\n\n")
                logging.info("===================")
                logging.info("|   STEP  {:5d}   |".format(i))
                logging.info("===================")
                logging.info("estimated validation accuracy {:.4f}".format(accuracy))
                logging.info("validation accuracy histogram: "+''.join(['({:.2f},{:.2f},  {:d}),'.format(100*bins[i],100*bins[i+1],cnts[i]) for i in range(len(cnts))]))
                logging.info("estimated accuracy loss: {:.4f}".format(accuracy_loss))
                logging.info("estimated diversity loss: {:.4f}".format(diversity_loss))
                logging.info("estimated entropy: {:.4f}".format(entropy))
                logging.info("estimated total loss: {:.4f}".format(np.mean(total_loss)))
                logging.info('majority vote ensemble accuracy: {:.4f}'.format(acc1))
                logging.info('maximum mean probabilities ensemble accuracy: {:.4f}'.format(acc2))
                logging.info('most confident individual ensemble accuracy: {:.4f}'.format(acc3))
                logging.info('learning rate: {:.3e}'.format(learning_rate))
                logging.info('learning rate rate: {:.6e}'.format(learning_rate_rate))
                logging.info('lambda: {:.3e}'.format(lamBda))
                logging.info('lambda rate: {:.6e}'.format(lambda_rate))
                logging.info("-----------------------------\n\n")
                hnet.SaveToCheckpoint(sess,CHECKPOINT_FILENAME)

            z = hnet.SampleInput(task='train')

            x, y = GetImages('train', hnet.generator_hparams.images_batch_size,hnet.generator_hparams.noise_batch_size)

            if i % 100 == 0:
                accuracy, accuracy_loss, diversity_loss, total_loss = hnet.GetMetrics(sess,[hnet.average_accuracy,hnet.accuracy_loss,hnet.diversity_loss,hnet.loss],x,y,z,is_training=True)
                logging.info("step {:d}: estimated accuracy >>>{:.4f}<<<".format(i, accuracy))
                logging.info(' (accuracy_loss, diversity_loss, total_loss): ({:.5f}, {:.5f} ,{:.5f})'.format(accuracy_loss, diversity_loss,total_loss))

                # check if there are new values for updating learning rate and lambda
                UpdateStuff(sess, hnet)

            i = hnet.TrainStep(sess, x, y,z)

if __name__=="__main__":
    hnet = HyperNetwork(use_generator=USE_GENERATOR)

    if USE_GENERATOR:
        print('number of parameters: {:d}'.format(int(hnet.NumberOfParameters())))
        TrainGenerator(hnet)
    else:
        TrainClassifier(hnet)
