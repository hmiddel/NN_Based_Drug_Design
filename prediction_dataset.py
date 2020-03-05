import re

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from Protein_embedding import embed_protein
from smiles_embedding import embed_single_smiles
from training_plot import plots


def prediction_dataset(embedded_smiles, embedded_prot):
    preds = []
    for i in range(5):
        model = tf.keras.models.load_model('data/model_save' + str(i))
        print("model loaded")
        prediction = []
        for j, prot in enumerate(embedded_prot):
            prot = tf.ragged.constant([prot]).to_tensor(shape=(1, None, 100))
            compound = tf.ragged.constant([embedded_smiles[j]]).to_tensor(shape=(1, None, 100))
            prediction.append(float(model.predict(x=[compound, prot])))
        preds.append(prediction)
    return preds


if __name__ == "__main__":
    data = pd.read_csv("data/BindingDB_IC50.tsv", sep="\t", usecols=["IC50", "SMILES embedding", "Protein embedding"],
                       dtype={"IC50": np.float64})
    sample(frac=0.01)
    digits = re.compile(r'[\d\.-]+')
    paragraph = re.compile(r"\[.+?\]")
    data["SMILES embedding"] = [[list(map(float, digits.findall(token)))
                                 for token in paragraph.findall(embedding)]
                                for embedding in data["SMILES embedding"]]
    data["Protein embedding"] = [[list(map(float, digits.findall(token)))
                                  for token in paragraph.findall(embedding)]
                                 for embedding in data["Protein embedding"]]
    pred = prediction_dataset(list(data["SMILES embedding"]), list(data["Protein embedding"]))
    for i in range(5):
        plots(pred[i], data["IC50"], i)
    pred = np.mean(pred, axis=1)
    plots(preddata["IC50"], 'mean')
