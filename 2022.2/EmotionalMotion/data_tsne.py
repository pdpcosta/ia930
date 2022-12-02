from process_data import load_and_extract
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set pyplot font size configs
plt.rc('font', size=20) 
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)

def load_data_per_subject(root_dir):

    mo_files_names = [name for name in os.listdir(root_dir + 'sensor_data/') if 'mo_' in name]
    mu_files_names = [name for name in os.listdir(root_dir + 'sensor_data/') if 'mu_' in name]
    mw_files_names = [name for name in os.listdir(root_dir + 'sensor_data/') if 'mw_' in name]

    mo_data = [load_and_extract(root_dir + 'sensor_data/', files_names=[subject], debug=False) for subject in mo_files_names]
    mu_data = [load_and_extract(root_dir + 'sensor_data/', files_names=[subject], debug=False) for subject in mu_files_names]
    mw_data = [load_and_extract(root_dir + 'sensor_data/', files_names=[subject], debug=False) for subject in mw_files_names]
    
    return mo_data, mu_data, mw_data


mo_data, mu_data, mw_data = load_data_per_subject('./data/')
print(len(mo_data), len(mu_data), len(mw_data))




def plot_embed_projection(features_vectors, samples_per_subject, p, ee, lr, n_iter, ax=None):

    features_vectors = StandardScaler().fit_transform(np.asarray(features_vectors))
    projection = TSNE(init='pca', perplexity=p, n_iter=n_iter, early_exaggeration=ee, learning_rate=lr).fit_transform(features_vectors)
    
    labels = []
    [labels.extend(['Subject ' + str(i+1)] * samples_per_subject[i]) for i in range(len(samples_per_subject))]

    df = pd.DataFrame({'x': [x for x in projection[:, 0]],
                        'y': [y for y in projection[:, 1]],
                    'word': labels})


    sns.scatterplot(data=df, x='x', y='y', hue='word', s=100, ax=ax, legend=False)


fig, axs = plt.subplots(3,3)
fig.set_size_inches(3*12, 3*12)

for rows in axs:
    for ax in rows:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

axs[0][0].set_title('Experiment 1') # MW
axs[0][1].set_title('Experiment 2') # MO
axs[0][2].set_title('Experiment 3') # MU

axs[0][0].set_ylabel('Positive Emotion')
axs[1][0].set_ylabel('Neutral Emotion')
axs[2][0].set_ylabel('Negative Emotion')

for emotion_label in [0,1,2]:

    samples_per_subject = [[sum(sub[:, -1] == emotion_label) for sub in data[:10]] for data in [mw_data, mo_data, mu_data]]

    mw_data_stack, mo_data_stack, mu_data_stack = np.vstack(mw_data[:10]), np.vstack(mo_data[:10]), np.vstack(mu_data[:10])
    mw_data_stack = mw_data_stack[mw_data_stack[:, -1] == emotion_label]
    mo_data_stack = mo_data_stack[mo_data_stack[:, -1] == emotion_label] 
    mu_data_stack = mu_data_stack[mu_data_stack[:, -1] == emotion_label]

    
    plot_embed_projection(mw_data_stack, samples_per_subject[0], 50, 12, 250, 5000, ax=axs[emotion_label][0])
    plot_embed_projection(mo_data_stack, samples_per_subject[1], 50, 12, 250, 5000, ax=axs[emotion_label][1])
    plot_embed_projection(mu_data_stack, samples_per_subject[2], 50, 12, 250, 5000, ax=axs[emotion_label][2])

plt.show()