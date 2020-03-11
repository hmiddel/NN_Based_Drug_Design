import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from Model_Main import show_figures, run_model
from training_plot import plots
from visualization import sd_filter_boolean

# Dataset settings
DATASET_FRACTION = 0.01
CROSS_VALIDATION_NUMBER = 5


def main():
    X = []
    prediction = []

    digits = re.compile(r'[\d\.-]+')
    paragraph = re.compile(r"\[.+?\]")

    # Load dataset
    data = pd.read_csv("data/DTI/BindingDB_IC50.tsv", sep="\t")

    # Suffle the needed data
    data = data.sample(frac=DATASET_FRACTION)

    # Data normalization and filtering
    data["IC50"] = np.log10(data["IC50"])
    filtered = sd_filter_boolean(data["IC50"], 3)
    data = data[filtered]
    mean = np.mean(data["IC50"])
    sd = np.std(data["IC50"])
    data["IC50"] = (data["IC50"] - mean) / sd


    # Visualization of IC50
    figure = plt.figure()
    figure.suptitle('Log10 of IC50', fontsize=16)
    figure = plt.hist(data["IC50"], color='blue', edgecolor='black',
                      bins=50)
    plt.savefig("data/DTI_IC50.png")

    # Convert embeddings from str to float
    data["SMILES embedding"] = [[list(map(float, digits.findall(token))) for token in paragraph.findall(embedding)] for
                                embedding in data["SMILES embedding"]]
    data["Protein embedding"] = [[list(map(float, digits.findall(token))) for token in paragraph.findall(embedding)] for
                                 embedding in data["Protein embedding"]]

    # Get all smiles, proteins and IC50 to make predictions and compare
    smiles = tf.ragged.constant(data["SMILES embedding"]).to_tensor(shape=(None, None, 100))
    prot = tf.ragged.constant(data["Protein embedding"]).to_tensor(shape=(None, None, 100))
    IC50 = data["IC50"]

    # Divide data according to the cross validation number
    data = np.array_split(data, CROSS_VALIDATION_NUMBER)

    # Run the model multiple times for cross validation
    for i in range(CROSS_VALIDATION_NUMBER):
        test_data = pd.DataFrame(data[i])
        dataset = data[:]
        del dataset[i]
        train_data = pd.concat(dataset)
        del dataset

        # Load the data
        train_IC, test_IC = np.array(train_data["IC50"]), np.array(test_data["IC50"])

        # Reshape the embedded arrays for use with tensorflow
        embedded_train_smiles = tf.ragged.constant(train_data["SMILES embedding"]).to_tensor(shape=(None, None, 100))
        embedded_test_smiles = tf.ragged.constant(test_data["SMILES embedding"]).to_tensor(shape=(None, None, 100))
        embedded_train_prot = tf.ragged.constant(train_data["Protein embedding"]).to_tensor(shape=(None, None, 100))
        embedded_test_prot = tf.ragged.constant(test_data["Protein embedding"]).to_tensor(shape=(None, None, 100))
        del train_data, test_data

        # Run the model
        hist, pred = run_model(embedded_train_smiles, train_IC, embedded_test_smiles, test_IC, smiles,
                               embedded_train_prot, embedded_test_prot, prot)
        X.append(hist)
        prediction.append(pred)
    # Gather the metrics and plot prediction comparison for each cross validation run
    metrics = {'loss': [], 'mae': [], 'mape': [], 'val_loss': [], 'val_mae': [], 'val_mape': []}
    for i in range(CROSS_VALIDATION_NUMBER):
        for j in metrics.keys():
            metrics[j].append(X[i].history[j])
        label = "DTI_" + str(i)
        plots(IC50, prediction[i], label, save=True)
    for j in metrics.keys():
        metrics[j] = np.mean(metrics[j], axis=0)

    # Ensembling of the cross validation runs
    prediction = np.mean(prediction, axis=0)
    label = "DTI_mean"
    plots(IC50, prediction, label, save=True)
    print(metrics)

    # Show the gathered metrics
    show_figures(metrics, "DTI_evolution")


if __name__ == '__main__':
    main()
