from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras import Input, Model
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from CustomLayers import BiLSTMSelfAttentionLayer
from Protein_embedding import embed_protein
from smiles_embedding import embed_smiles
from visualization import min_max_scale
from data_extraction import get_info

# General settings
BATCH_SIZE = 64
# Dataset settings
DATASET_FRACTION = 1
CROSS_VALIDATION_NUMBER = 5
# Protein embedding settings
PROT_EMBEDDING_DIM = 50
PROT_WORDS_LENGTH = 3
WINDOW_SIZE = 5
NEGATIVE_SIZE = 5
# Bi-LSTM Self-attention layer settings
da = 15
r = 10
LSTM_SIZE = 10
DROPOUT_RATE = 0
# Training settings
EPOCHS = 400


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
    print('Final MAE :', metrics["val_mae"][-1], 'Final MAPE :', metrics["val_mape"][-1])


def run_model(train_smiles, train_prot, train_IC, test_smiles, test_prot, test_IC):
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
    input_protein = Input(shape=(None, PROT_EMBEDDING_DIM,), name="protein")

    selfattention_smiles = BiLSTMSelfAttentionLayer(da, r, LSTM_SIZE, DROPOUT_RATE)(input_smiles)
    selfattention_protein = BiLSTMSelfAttentionLayer(da, r, LSTM_SIZE, DROPOUT_RATE)(input_protein)

    full = concatenate([selfattention_smiles, selfattention_protein])

    pred = Dense(1, activation="linear")(Dense(20, activation="tanh")(full))

    model = Model(
        inputs=[input_smiles, input_protein],
        outputs=pred
    )

    # utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True)

    model.compile(optimizer="adam",
                  loss="mse",
                  metrics=["mae", "mse"])

    return model.fit(x=[train_smiles, train_prot], y=train_IC,
                     validation_data=([test_smiles, test_prot], test_IC),
                     BATCH_SIZE=BATCH_SIZE,
                     EPOCHS=EPOCHS)


def main():
    X = []

    # Load first dataset, produced by Angel
    sdf_test = ["data/Angel_dataset/scorp/fixed-conformers_3d_scorp (" + str(j + 1) + ").sdf" for j in range(
        71)] + ["data/Angel_dataset/scorp/fixed-conformers_3d_3d_scorp (" + str(i + 1) + ").sdf" for i in range(23)]
    pdb_test = ["data/Angel_dataset/Protein/bs_protein (" + str(i + 1) + ").pdb" for i in range(94)]
    _, info = get_info(sdf_test, "data/Angel_dataset/pro.pdb", pdb_test)
    data_angel = pd.DataFrame(info)
    data_angel = data_angel.dropna()

    # Load second dataset, produced by Xin
    sdf_test = ["data/Xin_dataset/scorp/fixed-conformers_3d_scorp Xin(" + str(j + 1) + ").sdf" for j in range(
        71)] + ["data/Xin_dataset/scorp/fixed-conformers_3d_3d_scorp Xin(" + str(i + 1) + ").sdf" for i in range(23)]
    pdb_test = ["data/Xin_dataset/Protein/bs_protein Xin(" + str(i + 1) + ").pdb" for i in range(82)]
    _, info = get_info(sdf_test, "data/Xin_dataset/proXin.pdb", pdb_test)
    data_xin = pd.DataFrame(info)
    data_xin = data_xin.dropna()

    # Combine datasets
    data = pd.concat([data_angel, data_xin], axis=0)
    data = data.sample(frac=DATASET_FRACTION)
    data = np.array_split(data, CROSS_VALIDATION_NUMBER)

    # Run the model multiple times for cross validation
    for i in range(CROSS_VALIDATION_NUMBER):
        test_data = pd.DataFrame(data[i])
        dataset = data[:]
        del dataset[i]
        train_data = pd.concat(dataset)
        del dataset

        # Load the data and normalize it, if needed
        train_smiles, test_smiles = train_data['SMILES'], test_data['SMILES']
        train_prot, test_prot = train_data["Sequence"], test_data["Sequence"]
        train_IC, test_IC = np.array(min_max_scale(train_data["score"])), np.array(
            min_max_scale(test_data["score"]))

        del train_data, test_data

        # Embed the smiles and the proteins
        embedded_train_smiles, embedded_test_smiles = np.array(embed_smiles(train_smiles)), np.array(
            embed_smiles(test_smiles))
        embedded_train_prot, embedded_test_prot = embed_protein(PROT_EMBEDDING_DIM, train_prot, PROT_WORDS_LENGTH,
                                                                WINDOW_SIZE, NEGATIVE_SIZE), \
                                                  embed_protein(PROT_EMBEDDING_DIM, test_prot, PROT_WORDS_LENGTH,
                                                                WINDOW_SIZE, NEGATIVE_SIZE)

        # Reshape the embedded arrays for use with tensorflow
        embedded_train_smiles = tf.ragged.constant(embedded_train_smiles).to_tensor(shape=(None, None, 100))
        embedded_test_smiles = tf.ragged.constant(embedded_test_smiles).to_tensor(shape=(None, None, 100))
        embedded_train_prot = tf.ragged.constant(embedded_train_prot).to_tensor(shape=(None, None, PROT_EMBEDDING_DIM))
        embedded_test_prot = tf.ragged.constant(embedded_test_prot).to_tensor(shape=(None, None, PROT_EMBEDDING_DIM))

        # Run the model
        X.append(
            run_model(embedded_train_smiles, embedded_train_prot, train_IC, embedded_test_smiles, embedded_test_prot,
                      test_IC))

    # Gather the metrics for each cross validation run
    metrics = {'loss': [], 'mae': [], 'mse': [], 'val_loss': [], 'val_mae': [], 'val_mse': []}
    for i in range(CROSS_VALIDATION_NUMBER):
        for j in metrics.keys():
            metrics[j].append(X[i].history[j])
    for j in metrics.keys():
        metrics[j] = np.mean(metrics[j], axis=0)
    print(metrics)

    # Show the gathered metrics
    show_figures(metrics)


if __name__ == '__main__':
    main()
