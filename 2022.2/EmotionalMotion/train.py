from process_data import load_and_extract
import pandas as pd
import os
from joblib import dump, load
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

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
    (SVC(random_state=42), [{"kernel": ["poly"], "degree": [2, 3, 4, 5], "C": [ 10**i for i in range(1,4)]},
                            {"kernel": ["rbf", "sigmoid"], "C": [ 10**i for i in range(1,4)]}]),
    (LogisticRegression(random_state=42, max_iter=15000), {"C": [ 10**i for i in range(1,4)],
                                                           "solver": ["lbfgs", "liblinear", "sag", "saga"]}),
    (MLPClassifier(max_iter=5000), {'alpha': [10**(-i) for i in range(3,7)],
                                    'solver': ['sgd', 'adam'],
                                    'activation': ['tanh', 'relu'],
                                    'hidden_layer_sizes': [(10), (20), (50), (100), (10, 10), (20, 10),
                                                            (20, 20), (50, 20), (50, 50), (100, 20),
                                                            (100, 50), (100, 100), (50, 20, 10),
                                                            (50, 20, 20), (50, 50, 20), (100, 50, 20),
                                                            (100, 50, 50)]}),
    (RandomForestClassifier(), {'n_estimators': [20, 50, 100, 200],
                                'criterion': ['gini', 'entropy', 'log_loss']}),
    (AdaBoostClassifier(), {'n_estimators': [20, 50, 100, 200],
                            'learning_rate': [10**(-i) for i in range(4)]})
]
 
best_models = []
for model, params in models:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=0)
    search = GridSearchCV(estimator=model, param_grid=params, scoring="accuracy", cv=cv, verbose=3, n_jobs=1)
    search.fit(x_train, y_train)
    best_models.append(pd.DataFrame(search.cv_results_))

results_df = pd.concat(best_models, sort=False)
results_df["max_test_score"] = results_df[['split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score',]].max(axis=1)
results_df["min_test_score"] = results_df[['split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score',]].min(axis=1)
results_df = results_df.sort_values(by=["mean_test_score"])
print(results_df[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']])
results_df[['params', 'mean_test_score', 'std_test_score', 'max_test_score', 'min_test_score', 'rank_test_score']].to_csv('./data/grid_search_results.csv', index=False)
