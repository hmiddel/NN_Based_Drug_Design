import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, concatenate

from CustomLayers import BiLSTMSelfAttentionLayer
from training_plot import plots
from visualization import sd_filter_boolean

# General settings
BATCH_SIZE = 64
# Dataset settings
DATASET_FRACTION = 0.001
CROSS_VALIDATION_NUMBER = 5
# Bi-LSTM Self-attention layer settings
da = 15
r = 10
LSTM_SIZE = 10
DROPOUT_RATE = 0
# Training settings
EPOCHS = 10


def show_figures(metrics):
    """
    Shows figures for a list of metrics
    :param metrics: a list of metrics
    :return: None
    """
    fig = plt.figure(figsize=(8, 12))
    fig.suptitle('Metrics evoltion over EPOCHS', fontsize=18)
    ax1 = fig.add_subplot(311)
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss function')
    ax1.plot(metrics["loss"], '-b')
    ax1.plot(metrics["val_loss"], '-r')
    plt.legend(('Training loss', 'Validation loss'),
               loc='upper right')

    ax2 = fig.add_subplot(312)
    ax2.set_ylabel('MAE')
    ax2.set_title('MAE')
    ax2.plot(metrics["mae"], '-b')
    ax2.plot(metrics["val_mae"], '-r')
    plt.legend(('Training MAE', 'Validation MAE'),
               loc='upper right')

    ax3 = fig.add_subplot(313)
    ax3.set_ylabel('MAPE')
    ax3.set_xlabel('Epochs')
    ax3.set_title('MAPE')
    ax3.plot(metrics["mape"], '-b', label='MAPE')
    ax3.plot(metrics["val_mape"], '-r')
    plt.legend(('Training MAPE', 'Validation MAPE'),
               loc='upper right')
    plt.show()
    plt.savefig('data/evolution_scorp.png')
    print('Final MAE :', metrics["val_mae"][-1], 'Final MAPE :', metrics["val_mape"][-1])


def run_model_protein(train_smiles, train_prot, train_IC, test_smiles, test_prot, test_IC, smiles, prot):
    """
    Runs the actual model on the specified data
    :param train_smiles: the embedded training smiles
    :param train_prot: the embedded training proteins
    :param train_IC: the training IC50
    :param test_smiles: the embedded testing smiles
    :param test_prot: the embedded testing proteins
    :param test_IC: the testing IC50
    :return: the results of the model fit as a History object
    """
    input_smiles = Input(shape=(None, 100,), name="smiles")
    input_protein = Input(shape=(None, 100,), name="protein")

    selfattention_smiles = BiLSTMSelfAttentionLayer(da, r, LSTM_SIZE, DROPOUT_RATE)(input_smiles)
    selfattention_protein = BiLSTMSelfAttentionLayer(da, r, LSTM_SIZE, DROPOUT_RATE)(input_protein)

    full = concatenate([selfattention_smiles, selfattention_protein])

    pred = Dense(1, activation="linear")(Dense(16, activation="relu")(Dense(32, activation="relu")(full)))

    model = Model(
        inputs=[input_smiles, input_protein],
        outputs=pred
    )

    # utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True)

    model.compile(optimizer="adam",
                  loss="mse",
                  metrics=["mae", "mape"])

    X = model.fit(x=[train_smiles, train_prot], y=train_IC,
                  validation_data=([test_smiles, test_prot], test_IC),
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS)
    pred = model.predict([smiles, prot])
    return X, pred


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
    mean = np.mean(data["IC50"])
    sd = np.std(data["IC50"])
    data["IC50"] = (data["IC50"] - mean) / sd
    filter = sd_filter_boolean(data["IC50"], 3)
    data = data[filter]

    # Visualization of IC50
    figure = plt.figure()
    figure.suptitle('Log10 of IC50', fontsize=16)
    figure = plt.hist(data["IC50"], color='blue', edgecolor='black',
                      bins=100)
    plt.savefig("data/IC50.png")

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
        hist, pred = run_model_protein(embedded_train_smiles, embedded_train_prot, train_IC, embedded_test_smiles,
                                       embedded_test_prot, test_IC, smiles, prot)
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
    show_figures(metrics)


if __name__ == '__main__':
    main()
