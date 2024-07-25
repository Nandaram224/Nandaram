# Diabetes Prediction Using Deep Neural Network

This project uses a deep neural network to predict diabetes based on a dataset of medical parameters. The model is tuned using Keras Tuner to find the best hyperparameters for optimal performance.

### Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Building](#model-building)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Training and Evaluation](#training-and-evaluation)
8. [Results](#results)
9. [Conclusion](#conclusion)
10. [References](#references)

### Introduction

This project aims to build a deep learning model to predict the onset of diabetes using various medical parameters. The dataset used is the Pima Indians Diabetes Database, which contains information about female patients aged 21 and above.

### Dataset

The dataset is stored in a CSV file named `diabetes.csv`. It contains the following columns:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome

The `Outcome` column is the target variable, indicating whether the patient has diabetes (1) or not (0).

### Installation

To run this project, you'll need Python and the necessary libraries. You can install the required libraries using `pip`


pip install numpy pandas scikit-learn tensorflow keras keras-tuner

Data Preprocessing:
Load the dataset using Pandas.
Scale the features using StandardScaler from scikit-learn.
Split the dataset into training and testing sets.
Model Building
A sequential model is built using TensorFlow and Keras. The model architecture includes:

Input layer with 8 nodes (one for each feature).
One or more hidden layers with ReLU activation.
Output layer with sigmoid activation for binary classification.
Hyperparameter Tuning
Keras Tuner is used to find the optimal hyperparameters:

Number of units in each hidden layer.
Number of hidden layers.
Type of optimizer.
Training and Evaluation
The best model found by Keras Tuner is trained on the training data and evaluated on the testing data. The model is trained for 100 epochs with a batch size of 32.

Results
The performance of the model is evaluated based on accuracy and loss on the testing data.

### Conclusion

The project demonstrates the use of deep learning and hyperparameter tuning to predict diabetes. The tuned model provides a reliable method for predicting diabetes based on medical parameters.



import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

import keras_tuner as kt

# Load and preprocess data:

df = pd.read_csv('diabetes.csv')
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Build model function
def build_model(hp):

    model = Sequential()
    model.add(Dense(hp.Int('units', min_value=8, max_value=128, step=8), activation='relu', input_dim=8))
    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(Dense(hp.Int('units', min_value=8, max_value=128, step=8), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop', 'adagrad']),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model

# Hyperparameter tuning
tuner = kt.RandomSearch(
    
    build_model,
    objective='val_accuracy',
    max_trials=5,
    directory='my_dir',
    project_name='tuning_model'
)

tuner.search(x_train, y_train, validation_data=(x_test, y_test), epochs=5)

# Get the best model

best_model = tuner.get_best_models(num_models=1)[0]

best_model.summary()

# Train the best model
best_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=32)

# References
Pima Indians Diabetes Database

Keras Documentation

Keras Tuner Documentation
## 🚀 About Me
An Aspiring Data Scientist and machine learning enthusiast with a keen interest in natural language processing and AI applications. My background includes extensive experience in Python programming, data analysis, and developing machine learning models. I created this project to explore the intersection of text extraction, hierarchical indexing, and question-answering systems.

Through this project, I aim to provide an efficient and user-friendly way to extract, index, and retrieve relevant information from PDF documents using advanced NLP techniques. I believe in the power of open-source collaboration and am excited to see how this project can evolve with contributions from the community.

Connect with Me:
GitHub: https://github.com/Nandaram224/Nandaram
LinkedIn: https://www.linkedin.com/in/nanda-ram-aba320128/
Email: Nandaramlsj@gmail.com

