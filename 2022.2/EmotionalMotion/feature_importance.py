from process_data import load_and_extract
import os
from joblib import dump, load
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
from tqdm import tqdm

matplotlib.use('TKAgg')
# Set pyplot font size configs
plt.rc('font', size=20) 
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)

def load_data(root_dir):
    np.random.seed(0)

    data_dir = root_dir + 'extracted_data/'

    if os.path.exists(data_dir + 'train.csv') and os.path.exists(data_dir + 'test.csv'):
        train_data = np.loadtxt(data_dir + 'train.csv', delimiter=',')
        test_data = np.loadtxt(data_dir + 'test.csv', delimiter=',')

    else:
        data = load_and_extract(root_dir + 'sensor_data/')
        np.random.shuffle(data)

        split_idx = int(np.ceil(len(data) * 0.8))
        print(len(data), split_idx)
        train_data = data[:split_idx]
        test_data = data[split_idx:]


        np.savetxt(data_dir + 'train.csv', train_data, fmt='%f', delimiter=',')
        np.savetxt(data_dir + 'test.csv', test_data, fmt='%f',delimiter=',')

    return train_data, test_data

train_data, test_data = load_data('./data/')
x_train, y_train = train_data[:, :-1], train_data[:, -1]
x_test, y_test = test_data[:, :-1], test_data[:, -1]

main_clf = load('best_clf.joblib')

feature_per_axis = ['mean', 'median','max','min','range','qurtile range','variance','std_dev','mean_abs_dev','assimetry','kurtoisis','rms','energy','potency','zero_cross','inclination', 'fft_1','fft_2','fft_3','fft_4','fft_5','fft_6','fft_7','fft_8','fft_9','fft_10','fft_11','fft_12','fft_13','fft_14','fft_15','fft_mean','fft_energy']
features_names = []
[features_names.append(axis + '__' + feat) for axis in ['acc_x', 'acc_y', 'acc_z'] for feat in feature_per_axis]
features_names.extend(['acc__mag', 'acc_area', 'acc_mean_jerk', 'acc_std_jerk', 'acc_angle_x', 'acc_angle_y', 'acc_angle_z'])
[features_names.append(axis + '__' + feat) for axis in ['gyr_x', 'gyr_y', 'gyr_z'] for feat in feature_per_axis]
features_names.extend(['gyr__mag', 'gyr_area', 'gyr_mean_jerk', 'gyr_std_jerk', 'gyr_angle_x', 'gyr_angle_y', 'gyr_angle_z'])

all_tests = None

for _ in tqdm(range(50)):
    permute_clf = RandomForestClassifier(n_estimators=200, criterion='entropy')
    permute_clf.fit(x_train, y_train)

    result = permutation_importance(
        permute_clf, x_test, y_test, n_repeats=5, n_jobs=-1
    )

    if all_tests is None:
        all_tests = result.importances_mean
    else:
        all_tests = all_tests + result.importances_mean

forest_importances = pd.Series(all_tests / 50, index=features_names)

fig, ax = plt.subplots()
fig.set_size_inches(150,25)
forest_importances.sort_values(ascending=False).plot.bar(ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.savefig('./figures/feature_importance.png')
plt.show()

#---- Training model with only positivie importance features------

feature_idxs = np.where((forest_importances > 0).to_numpy())[0]
x_train_importance = x_train[:, feature_idxs]
x_test_importance = x_test[:, feature_idxs]

importance_clf = RandomForestClassifier(n_estimators=200, criterion='entropy')
importance_clf.fit(x_train_importance, y_train)

y_pred = importance_clf.predict(x_test_importance)
y_pred_probs = importance_clf.predict_proba(x_test_importance)

print(f'Accuracy Score: {accuracy_score(y_test, y_pred)*100:.2f}%')
print(f'Balanced Accuracy Score: {balanced_accuracy_score(y_test, y_pred)*100:.2f}%')
f1 = f1_score(y_test, y_pred, average='macro')
print(f'F1 Score: {f1:.4f}')

cm = confusion_matrix(y_test, y_pred, normalize='true')
cmd = ConfusionMatrixDisplay(cm)

fig, ax = plt.subplots()
fig.set_size_inches(15,10)
ax.set_title('Important Features Classifier Confusion Matrix')

cmd.plot(ax=ax, cmap='GnBu')
plt.savefig('./figures/importance_model.png')
plt.show()
