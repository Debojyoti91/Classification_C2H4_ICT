# models_training.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def train_models(datafile_path):
    data = pd.read_csv(datafile_path)
    data = data[data['b'] <= 6.5]

    # Data Preprocessing
    data_process = data.copy()
    data_process.fillna(data_process.mode().iloc[0], inplace=True)
    data_process["react_type"].replace({"1": "I", "2": "II"}, inplace=True)
    X_data = data_process[['Alpha', 'Beta', 'Gamma', 'b']]
    y_data = data_process[['react_type']].values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.20, stratify=y_data, random_state=0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    ros = RandomOverSampler(random_state=0)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

    # Define the models and their parameter grids
    models = {
        'SVC': (SVC(), {'C': np.arange(0.1, 2, 0.5), 'kernel': ['linear', 'rbf']}),
        'DecisionTree': (DecisionTreeClassifier(), {'max_depth': np.arange(3, 6)}),
        'RandomForest': (RandomForestClassifier(), {'n_estimators': np.arange(50, 151, 50), 'max_depth': np.arange(3, 6)}),
        'KNN': (KNeighborsClassifier(), {'n_neighbors': np.arange(3, 6)})
    }

    results = {}
    for model_name, (model, param_grid) in models.items():
        clf = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        clf.fit(X_train_res, y_train_res)
        
        best_model = clf.best_estimator_
        y_pred = best_model.predict(X_test)

        best_score = clf.best_score_
        best_param = clf.best_params_
        best_classification = classification_report(y_test, y_pred)
        best_precision = precision_score(y_test, y_pred, average='weighted')
        best_recall = recall_score(y_test, y_pred, average='weighted')
        best_f1 = f1_score(y_test, y_pred, average='weighted')

        results[model_name] = {
            'best_score': best_score,
            'best_param': best_param,
            'best_classification': best_classification,
            'best_precision': best_precision,
            'best_recall': best_recall,
            'best_f1': best_f1
        }

    return results, X_test, y_test

if __name__ == '__main__':
    datafile = "../data/c2h4_final_data_phi_1.csv"
    train_models(datafile)

