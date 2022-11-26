from process_data import load_and_extract
import os
from joblib import dump, load
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

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

print(len(x_train), len(x_test))

if not os.path.exists('./best_clf.joblib'):
    clf = RandomForestClassifier(n_estimators=200, criterion='entropy')
    clf.fit(x_train, y_train)
    dump(clf, 'best_clf.joblib')

clf = load('best_clf.joblib')

y_pred = clf.predict(x_test)
y_pred_probs = clf.predict_proba(x_test)

print(f'Accuracy Score: {accuracy_score(y_test, y_pred)*100:.2f}%')
print(f'Accuracy Score: {balanced_accuracy_score(y_test, y_pred)*100:.2f}%')
print(f1_score(y_test, y_pred, average='macro'))
cm = confusion_matrix(y_test, y_pred, normalize='true')
print(cm)
cmd = ConfusionMatrixDisplay(cm).plot()