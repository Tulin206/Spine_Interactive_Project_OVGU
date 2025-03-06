import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import KFold, cross_validate, GridSearchCV, ShuffleSplit, train_test_split, learning_curve
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score


### SUPERVISED APPROACH - CLASSIFICATION MODELS
# SVM Model
def grid_search(model, param_grid, X, y):
    grid = GridSearchCV(model, param_grid=param_grid)
    grid.fit(X, y)
    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))
    return grid.best_params_


def evaluation(model, X, y):
    ss = ShuffleSplit(n_splits=10, test_size=.25, random_state=0)
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1_score': make_scorer(f1_score)}

    results = cross_validate(model, X, y, cv=ss, scoring=scoring)
    print(results.keys())
    print('Test Accuracy:', np.mean(results['test_accuracy']))
    print('Test Precision:', np.mean(results['test_precision']))
    print('Test Recall:', np.mean(results['test_recall']))
    print('Test F1 Score:', np.mean(results['test_f1_score']))


def model_svm(X, y):
    svc = SVC()
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(C=C_range, gamma_range=gamma_range)
    # find the best parameters for the SVC model
    best_params = grid_search(svc, param_grid, X, y)
    optimized_svc = SVC(best_params)
    evaluation(optimized_svc, X, y)


def model_random_forest(X, y):
    rf = RandomForestClassifier()
    param_grid = {'max_depth': range(3, 10, 2),
                  'max_features': ['auto', 'sqrt', 'log2'],
                  'criterion': ['gini', 'entropy']}
    best_params = grid_search(rf, param_grid, X, y)
    optimized_rf = RandomForestClassifier(best_params)
    evaluation(optimized_rf, X, y)


def model_mlp(X, y):
    mlp = MLPClassifier(max_iter=1000)
    param_grid = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        'activation': ['tanh', 'relu', 'logistic'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init': [0.5, 0.1, 0.001]
    }
    best_params = grid_search(mlp, param_grid, X, y)
    optimized_mlp = MLPClassifier(best_params)
    evaluation(optimized_mlp, X, y)

### UNSUPERVISED APPROACH - Explainability açısından nasıl?
# One class SVM
