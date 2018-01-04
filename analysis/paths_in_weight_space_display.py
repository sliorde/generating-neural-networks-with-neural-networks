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

# open file created with path_data_creator.py
with open('path_mnf.pickle','rb') as f:
    data = pickle.load(f)

direct = data['accuracy_direct']
non_direct = data['accuracy_interp']
zs = data['z']

t = np.linspace(0, 1, len(direct[0]))

cmap = plt.get_cmap('Dark2', 2)

for i in range(len(direct)):
    fig = plt.figure(figsize=(1.2, 1), facecolor='white')
    ax = plt.axes()

    ax.set_color_cycle([cmap(i) for i in range(2)])
    plt.plot(t,direct[i]*100,'--',lw=1.0,label='direct path')
    plt.plot(t,non_direct[i]*100,lw=1.0,label='interpolated path')
    plt.xlabel('t')
    plt.ylabel('accuracy [%]')
    ax.get_xaxis().set_ticks(np.arange(0,1.01,0.5))
    ax.get_yaxis().set_ticks(np.arange(0,104,5))
    plt.ylim(np.min(direct[i]*100)-5,108)
    # ax.legend(loc='lower center',labelspacing=0.1,borderpad=0.2)
    plt.gca().set_ylim(top=105)
    plt.tight_layout(0,0,0)