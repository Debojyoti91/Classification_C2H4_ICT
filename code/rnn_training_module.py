import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import random as rn

def train_c2h4_model(data_path):

    np.random.seed(1234)
    rn.seed(1254)
    tf.random.set_seed(1234)

    # Load data
    data = pd.read_csv(data_path)
    data = data[data['b'] <= 6.5]
    data_process = data.copy()

    # Data preprocessing
    data_process.fillna(data_process.mode().iloc[0], inplace=True)  # Mode imputation
    data_process["react_type"].replace({"1": "I", "2": "II"}, inplace=True)

    # Train-test splitting
    X_data = data_process[['Alpha', 'Beta', 'Gamma', 'b']]
    y_data = data_process[['react_type']]

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.20, stratify=y_data, random_state=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = y_train.values.ravel()

    ros = RandomOverSampler(random_state=0)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
    
    # One-hot encode the target variables and get the mapping
    num_classes = len(np.unique(y_train_res))
    y_train_res_encoded = pd.get_dummies(y_train_res)
    
    y_train_res = y_train_res_encoded.values
    y_test_res_encoded = pd.get_dummies(y_test)
    y_test_res = y_test_res_encoded.values

    X_train_res = X_train_res.reshape(-1, 4, 1)
    X_test = X_test.reshape(-1, 4, 1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.SimpleRNN(64, input_shape=(4, 1), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(tf.keras.optimizers.Adam(lr=0.0001), "categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(X_train_res, y_train_res, epochs=250, batch_size=32, verbose=1,
                        validation_data=(X_test, y_test_res))

    return model, scaler, class_mapping


