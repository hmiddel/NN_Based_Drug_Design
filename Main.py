from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras import Input, Model
import pandas as pd
import sklearn
import numpy as np
import tensorflow as tf

from CustomLayers import BiLSTMSelfAttentionLayer
from Protein_embedding import embed_protein
from smiles_embedding import embed_smiles
from visualization import min_max_scale

if __name__ == '__main__':
    # General settings
    batch_size = 64
    # Dataset settings
    test_size = 0.005
    train_size = 0.02
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

    data = pd.read_csv("data/binding_data_final.tsv", sep="\t")
    train_data, test_data = sklearn.model_selection.train_test_split(data, test_size=test_size, train_size=train_size)
    del data
    train_smiles, test_smiles = train_data['Ligand SMILES'], test_data['Ligand SMILES']
    train_prot, test_prot = train_data["BindingDB Target Chain  Sequence"], test_data[
        "BindingDB Target Chain  Sequence"]
    train_IC, test_IC = np.array(min_max_scale(train_data["IC50 (nm)"])), np.array(
        min_max_scale(test_data["IC50 (nm)"]))

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

    model.fit(x=[embedded_train_smiles, embedded_train_prot], y=train_IC,
              validation_data=([embedded_test_smiles, embedded_test_prot], test_IC),
              batch_size=batch_size,
              epochs=epochs)
