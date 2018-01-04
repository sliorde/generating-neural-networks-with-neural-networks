import numpy as np
import pickle

# open file created with accuracies_data_creator.py
with open('accuracies.pickle','rb') as f:
    data = pickle.load(f)

print('ensemble mean accuracy: {:.4f}'.format(np.mean(data['acc1'])))