import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, concatenate

from CustomLayers import BiLSTMSelfAttentionLayer
from visualization import sd_filter_boolean

# General settings
BATCH_SIZE = 64
# Dataset settings
DATASET_FRACTION = 0.01
# Bi-LSTM Self-attention layer settings
da_range = [5, 10, 15, 20]
r_range = [5, 10, 15, 20]
LSTM_SIZE = [5, 10, 15, 20]
DROPOUT_RATE = [0, 0.1, 0.2]
# Training settings
EPOCHS = 5


def run_model_opt(train_smiles, train_prot, train_IC, test_smiles, test_prot, test_IC, da, r, LSTM_SIZE, DROPOUT):
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

    selfattention_smiles = BiLSTMSelfAttentionLayer(da, r, LSTM_SIZE, DROPOUT)(input_smiles)
    selfattention_protein = BiLSTMSelfAttentionLayer(da, r, LSTM_SIZE, DROPOUT)(input_protein)

    full = concatenate([selfattention_smiles, selfattention_protein])

    pred = Dense(1, activation="linear")(Dense(20, activation="relu")(full))

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
    model.save("data/model_relu_LSTM_DROPOUT_r_da" + str(LSTM_SIZE) + str(DROPOUT) + str(r) + str(da))
    return X


def main_opt():
    X = []

    digits = re.compile(r'[\d\.-]+')
    paragraph = re.compile(r"\[.+?\]")

    # Load dataset
    data = pd.read_csv("data/BindingDB_IC50.tsv", sep="\t")

    # Suffle the needed data
    data = data.sample(frac=DATASET_FRACTION)

    # Log scale and filter IC50
    data["IC50"] = np.log10(data["IC50"])
    filter = sd_filter_boolean(data["IC50"], 3)
    data = data[filter]

    # Visualization of IC50
    figure = plt.figure()
    figure.suptitle('Log10 of IC50', fontsize=16)
    figure = plt.hist(data["IC50"], color='blue', edgecolor='black',
                      bins=100)
    plt.savefig("data/IC50_HPopt.png")

    # Convert embeddings from str to float
    data["SMILES embedding"] = [[list(map(float, digits.findall(token))) for token in paragraph.findall(embedding)] for
                                embedding in data["SMILES embedding"]]
    data["Protein embedding"] = [[list(map(float, digits.findall(token))) for token in paragraph.findall(embedding)] for
                                 embedding in data["Protein embedding"]]
    print("converted to float")
    print(data["SMILES embedding"])

    data = np.array_split(data, 5)
    print("data loaded")

    test_data = pd.DataFrame(data[0])
    dataset = data[:]
    del dataset[0]
    train_data = pd.concat(dataset)
    del dataset

    # Load the data and normalize it, if needed
    train_IC, test_IC = np.array(train_data["IC50"]), np.array(test_data["IC50"])

    # Reshape the embedded arrays for use with tensorflow
    embedded_train_smiles = tf.ragged.constant(train_data["SMILES embedding"]).to_tensor(shape=(None, None, 100))
    embedded_test_smiles = tf.ragged.constant(test_data["SMILES embedding"]).to_tensor(shape=(None, None, 100))
    embedded_train_prot = tf.ragged.constant(train_data["Protein embedding"]).to_tensor(shape=(None, None, 100))
    embedded_test_prot = tf.ragged.constant(test_data["Protein embedding"]).to_tensor(shape=(None, None, 100))
    del train_data, test_data
    print("data distributed")

    # Run the model
    for da in da_range:
        for r in r_range:
            for lstm in LSTM_SIZE:
                for dropout in DROPOUT_RATE:
                    X.append(run_model_opt(embedded_train_smiles, embedded_train_prot, train_IC, embedded_test_smiles,
                                           embedded_test_prot, test_IC, da, r, lstm, dropout))

    # Gather the metrics for each cross validation run
    metrics = {'loss': [], 'mae': [], 'mape': [], 'val_loss': [], 'val_mae': [], 'val_mape': []}
    for i in range(len(X)):
        for j in metrics.keys():
            metrics[j].append(X[i].history[j][-1])
    print(metrics)

    l_da = len(da_range)
    l_r = len(r_range)
    l_lstm = len(LSTM_SIZE)
    l_dropout = len(DROPOUT_RATE)

    min_mse, idx_mse = min((val, idx) for (idx, val) in enumerate(metrics['val_loss']))
    dropout_mse = DROPOUT_RATE[idx_mse % l_dropout]
    lstm_mse = LSTM_SIZE[idx_mse // (l_dropout) % l_lstm]
    r_mse = r_range[idx_mse // (l_dropout * l_lstm) % l_r]
    da_mse = da_range[idx_mse // (l_dropout * l_lstm * l_da)]

    min_mae, idx_mae = min((val, idx) for (idx, val) in enumerate(metrics['val_mae']))
    dropout_mae = DROPOUT_RATE[idx_mae % l_dropout]
    lstm_mae = LSTM_SIZE[idx_mae // (l_dropout) % l_lstm]
    r_mae = r_range[idx_mae // (l_dropout * l_lstm) % l_r]
    da_mae = da_range[idx_mae // (l_dropout * l_lstm * l_da)]

    min_mape, idx_mape = min((val, idx) for (idx, val) in enumerate(metrics['val_mape']))
    dropout_mape = DROPOUT_RATE[idx_mape % l_dropout]
    lstm_mape = LSTM_SIZE[idx_mape // (l_dropout) % l_lstm]
    r_mape = r_range[idx_mape // (l_dropout * l_lstm) % l_r]
    da_mape = da_range[(idx_mape // (l_dropout * l_lstm * l_da))]

    print("Minimum MSE", min_mse, "for parameters", idx_mse, "Dropout=", dropout_mse, "LSTM=", lstm_mse, "r=", r_mse,
          "da=", da_mse, "\n",
          "Minimum MAE", min_mae, "for parameters", idx_mae, "Dropout=", dropout_mae, "LSTM=", lstm_mae, "r=", r_mae,
          "da=", da_mae, "\n",
          "Minimum MAPE", min_mape, "for parameters", idx_mape, "Dropout=", dropout_mape, "LSTM=", lstm_mape, "r=",
          r_mape, "da=", da_mape)


if __name__ == '__main__':
    main_opt()
