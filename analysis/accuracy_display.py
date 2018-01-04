import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rc('font', size=8)
plt.rc('axes', titlesize=8)
plt.rc('axes', labelsize=8)
plt.rc('axes', linewidth=0.6)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('legend', fontsize=8)
plt.rc('figure', titlesize=8)

# open file created with accuracies_data_creator.py
with open('accuracies_mnf.pickle','rb') as f:
    data = pickle.load(f)
acc = data['accs_per_z'].flatten()



fig = plt.figure(figsize=(1.53, 1.2),facecolor='white')
ax = plt.axes()
ax.hist(100*acc,np.arange(95,100,0.05),color='skyblue',ec='blue',linewidth=0.4)
plt.xlabel('accuracy [%]')
plt.ylabel('counts',labelpad=0.3)
ax.get_xaxis().set_ticks(np.arange(97,100,0.5))
ax.get_yaxis().set_ticks(np.arange(0,900,200))
plt.xlim(97,99.1)
plt.tight_layout(0,0,0)
plt.show()


