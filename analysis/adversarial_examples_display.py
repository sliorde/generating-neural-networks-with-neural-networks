import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['patch.linewidth'] = 0.6
plt.rc('font', size=8)
plt.rc('axes', titlesize=8)
plt.rc('axes', labelsize=8)
plt.rc('axes', linewidth=0.6)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('legend', fontsize=8)
plt.rc('figure', titlesize=8)


# open file created with adversarial_data_creator.py
with open('data_adversarial.pickle','rb') as f:
    data = pickle.load(f)

perturbations = data['perturbations']
cdf_single = data['cdf_single1']
cdf_ensemble = data['cdf_ensemble1']

fig = plt.figure(figsize=(2.1, 1.4), facecolor='white')
ax = plt.axes()
plt.plot(perturbations,cdf_single,lw=1.0,c=(77/255,153/255,136/255),label='single classifier')
plt.plot(perturbations,cdf_ensemble,lw=1.0,c=(176/255,86/255,26/255),label='ensemble')
plt.xlabel('perturbation size')
plt.ylabel('success probability')
plt.ylim(0,1)
ax.legend(loc='best',labelspacing=0.1,borderpad=0.2)
plt.tight_layout(0,0,0)

plt.show()