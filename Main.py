from tensorflow.keras.layers import Dense, Flatten, Conv2D, Bidirectional, LSTM, Softmax, Layer, concatenate
from tensorflow.keras import Input, Model, utils
from CustomLayers import BiLSTMSelfAttentionLayer
import pandas as pd
import sklearn

from Protein_embedding import word2vec
from smiles_embedding import embed_smiles

if __name__ == '__main__':
    batch_size = 64

    data = pd.read_csv("data/binding_data_cleared2.tsv", sep="\t")
    train_data, test_data = sklearn.model_selection.train_test_split(data, test_size=0.2, train_size=0.8)
    train_smiles, test_smiles = train_data['Ligand SMILES'], test_data['Ligand SMILES']
    train_prot, test_prot = train_data["BindingDB Target Chain  Sequence"], test_data[
        "BindingDB Target Chain  Sequence"]
    train_IC, test_IC = train_data["IC50 (nm)"], test_data["IC50 (nm)"]

    embedded_train_smiles, embedded_test_smiles = embed_smiles(train_smiles), embed_smiles(test_smiles)
    embedded_train_prot, embedded_test_prot = word2vec(100, train_prot, 3, 5, 5), word2vec(100, test_prot, 3, 5, 5)

    input_smiles = Input(shape=(None, 100,), name="smiles")
    input_protein = Input(shape=(None, 100,), name="protein")

    selfattention_smiles = BiLSTMSelfAttentionLayer(15, 10, 10)(input_smiles)
    selfattention_protein = BiLSTMSelfAttentionLayer(15, 10, 10)(input_protein)

    full = concatenate([selfattention_smiles, selfattention_protein])

    pred = Dense(1)(Dense(20, activation="tanh")(full))

    model = Model(
        inputs=[input_smiles, input_protein],
        outputs=pred
    )

    # utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True)

    model.compile(optimizer="adam",
                  loss="mse",
                  metrics=["mae", "mse"])

    model.fit([embedded_train_smiles, embedded_train_prot], train_IC,
              validation_data=([embedded_test_smiles, embedded_test_prot], test_IC),
              batch_size=batch_size,
              epochs=3)
