from process_data import load_and_extract
import os
from joblib import dump, load
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance

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


#---Training model with all data except one subject and testing on the left out subject -----

def load_data_leave_one_out(root_dir):

    train_files_names = [name for name in os.listdir(root_dir + 'sensor_data/') if 'ew44' not in name and 'mw_' in name]
    test_files_names = [name for name in os.listdir(root_dir + 'sensor_data/') if 'mw_ew44' in name]

    train_data = load_and_extract(root_dir + 'sensor_data/', files_names=train_files_names)
    test_data = load_and_extract(root_dir + 'sensor_data/', files_names=test_files_names)
    return train_data, test_data

train_data, test_data = load_data_leave_one_out('./data/')
x_train, y_train = train_data[:, :-1], train_data[:, -1]
x_test, y_test = test_data[:, :-1], test_data[:, -1]

print(len(x_train), len(x_test))


one_out_clf = RandomForestClassifier(n_estimators=200, criterion='entropy')
one_out_clf.fit(x_train, y_train)

y_pred = one_out_clf.predict(x_test)
y_pred_probs = one_out_clf.predict_proba(x_test)

print(f'Accuracy Score: {accuracy_score(y_test, y_pred)*100:.2f}%')
print(f'Balanced Accuracy Score: {balanced_accuracy_score(y_test, y_pred)*100:.2f}%')
f1 = f1_score(y_test, y_pred, average='macro')
print(f'F1 Score: {f1:.2f}')

cm = confusion_matrix(y_test, y_pred, normalize='true')
cmd = ConfusionMatrixDisplay(cm, display_labels=['Negative', 'Neutral', 'Positive'])

fig, ax = plt.subplots()
fig.set_size_inches(15,10)
ax.set_title('Leave One Out Classifier Confusion Matrix')

cmd.plot(ax=ax, cmap='GnBu')
plt.savefig('./figures/leave_one_out_cm.png')
plt.show()
