from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras import Input, Model
import pandas as pd
import numpy as np
import tensorflow as tf

from CustomLayers import BiLSTMSelfAttentionLayer
from Protein_embedding import embed_protein
from smiles_embedding import embed_smiles
from visualization import min_max_scale
from data_extraction import get_info

if __name__ == '__main__':
    # General settings
    batch_size = 64
    # Dataset settings
    dataset_fraction = 1
    cross_validation_number = 5
    # Protein embedding settings
    prot_embedding_dim = 50
    prot_words_length = 3
    window_size = 5
    negative_size = 5
    # Bi-LSTM Self-attention layer settings
    da = 15
    r = 10
    LSTM_size = 10
    dropout_rate = 0
    # Training settings
    epochs = 3

    X = []
    sdf_test = ["data/Angel_dataset/scorp/fixed-conformers_3d_scorp (" + str(j + 1) + ").sdf" for j in range(
        71)]  + ["data/Angel_dataset/scorp/fixed-conformers_3d_3d_scorp (" + str(i+1) + ").sdf" for i in range(23)]
    pdb_test = ["data/Angel_dataset/Protein/bs_protein (" + str(i + 1) + ").pdb" for i in range(94)]
    defect, info = get_info(sdf_test, "data/Angel_dataset/pro.pdb", pdb_test)
    data_angel = pd.DataFrame(info)
    data_angel = data_angel.dropna()

    sdf_test = ["data/Xin_dataset/scorp/fixed-conformers_3d_scorp (" + str(j + 1) + ").sdf" for j in range(
        71)] + ["data/Xin_dataset/scorp/fixed-conformers_3d_3d_scorp (" + str(i + 1) + ").sdf" for i in range(23)]
    pdb_test = ["data/Xin_dataset/Protein/bs_protein (" + str(i + 1) + ").pdb" for i in range(82)]
    defect, info = get_info(sdf_test, "data/Xin_dataset/pro.pdb", pdb_test)
    data_xin = pd.DataFrame(info)
    data_xin = data_angel.dropna()
    data = data_angel.append(data_angel)

    data = data.sample(frac=dataset_fraction)
    data = np.array_split(data, cross_validation_number)


    for i in range(cross_validation_number) :
        test_data = pd.DataFrame(data[i])
        dataset = data[:]
        del dataset[i]
        train_data = pd.concat(dataset)
        del dataset
        train_smiles, test_smiles = train_data['SMILES'], test_data['SMILES']
        train_prot, test_prot = train_data["Sequence"], test_data["Sequence"]
        train_IC, test_IC = np.array(min_max_scale(train_data["score"])), np.array(
            min_max_scale(test_data["score"]))

        del train_data, test_data

        embedded_train_smiles, embedded_test_smiles = np.array(embed_smiles(train_smiles)), np.array(
            embed_smiles(test_smiles))
        embedded_train_prot, embedded_test_prot = embed_protein(prot_embedding_dim, train_prot, prot_words_length,
                                                                window_size, negative_size), embed_protein(
            prot_embedding_dim, test_prot, prot_words_length, window_size, negative_size)

        embedded_train_smiles = tf.ragged.constant(embedded_train_smiles).to_tensor(shape=(None, None, 100))
        embedded_test_smiles = tf.ragged.constant(embedded_test_smiles).to_tensor(shape=(None, None, 100))
        embedded_train_prot = tf.ragged.constant(embedded_train_prot).to_tensor(shape=(None, None, prot_embedding_dim))
        embedded_test_prot = tf.ragged.constant(embedded_test_prot).to_tensor(shape=(None, None, prot_embedding_dim))

        input_smiles = Input(shape=(None, 100,), name="smiles")
        input_protein = Input(shape=(None, prot_embedding_dim,), name="protein")

        selfattention_smiles = BiLSTMSelfAttentionLayer(da, r, LSTM_size, dropout_rate)(input_smiles)
        selfattention_protein = BiLSTMSelfAttentionLayer(da, r, LSTM_size, dropout_rate)(input_protein)

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

        X.append(model.fit(x=[embedded_train_smiles, embedded_train_prot], y=train_IC,
                  validation_data=([embedded_test_smiles, embedded_test_prot], test_IC),
                  batch_size=batch_size,
                  epochs=epochs))


    metrics = {'loss': [], 'mae': [], 'mse': [], 'val_loss': [], 'val_mae': [], 'val_mse': []}
    for i in range(cross_validation_number):
        for j in metrics.keys():
            metrics[j].append(X[i].history[j])
    for j in metrics.keys():
        metrics[j] = np.mean(metrics[j], axis=0)
    print(metrics)