from process_data import load_and_extract
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

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

models = [
    (SVC(random_state=42), [{"kernel": ["poly"], "degree": [2, 3, 4], "C": [ 10**i for i in range(2)]},
                            {"kernel": ["rbf"], "C": [ 10**i for i in range(2)]}]),
    (LogisticRegression(random_state=42, max_iter=15000), {"C": [ 10**i for i in range(3)],
                                                           "solver": ["lbfgs", "liblinear", "sag", "saga"]})
]

best_models = []
for model, params in models:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=0)

    search = GridSearchCV(estimator=model, param_grid=params, scoring="accuracy", cv=cv, verbose=3, n_jobs=1)
    search.fit(x_train, y_train)

    best_models.append(pd.DataFrame(search.cv_results_))

results_df = pd.DataFrame(best_models)
results_df = results_df.sort_values(by=["rank_test_score"])

print(results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]])




#best_models = []
#params = {"kernel": ["linear", "rbf"]}

#cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=0)
#model = SVC(random_state=42)
#search = GridSearchCV(estimator=model, param_grid=params, scoring="accuracy", cv=3, verbose=10, n_jobs=1)
#search.fit(x_train[:100], y_train[:100])
#best_models.append(search)

