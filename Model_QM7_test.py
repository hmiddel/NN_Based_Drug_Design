import numpy as np
import pandas as pd
import tensorflow as tf

from Model_Main import show_figures, run_model
from smiles_embedding import embed_smiles
from training_plot import plots
from simple_model import run_simple_model

# Dataset settings
CROSS_VALIDATION_NUMBER = 5


def main():
    data = pd.read_csv("chemprop-master/data/qm7.csv")
    data = data.sample(frac=1)
    scores = list(data["u0_atom"])
    smiles = embed_smiles(data["smiles"])
    smiles = tf.ragged.constant(smiles).to_tensor(shape=(None, None, 100))
    data = np.array_split(data, CROSS_VALIDATION_NUMBER)
    X = []
    prediction = []

    for i in range(CROSS_VALIDATION_NUMBER):
        test_data = pd.DataFrame(data[i])
        dataset = data[:]
        del dataset[i]
        train_data = pd.concat(dataset)
        del dataset

        # Load the data
        train_smiles, test_smiles = embed_smiles(train_data['smiles']), embed_smiles(test_data['smiles'])
        embedded_train_smiles = tf.ragged.constant(train_smiles).to_tensor(shape=(None, None, 100))
        embedded_test_smiles = tf.ragged.constant(test_smiles).to_tensor(shape=(None, None, 100))
        train_IC, test_IC = np.array(train_data["u0_atom"]), np.array(test_data["u0_atom"])
        del train_data, test_data

        # Run the model
        hist, pred = run_simple_model(embedded_train_smiles, train_IC, embedded_test_smiles, test_IC, smiles)
        X.append(hist)
        prediction.append(pred)

    # Gather the metrics and plot prediction comparison for each cross validation run
    metrics = {'loss': [], 'mae': [], 'mape': [], 'val_loss': [], 'val_mae': [], 'val_mape': []}
    for i in range(CROSS_VALIDATION_NUMBER):
        for j in metrics.keys():
            metrics[j].append(X[i].history[j])
        label = 'QM7_' + str(i)
        plots(scores, prediction[i], label, save=True)

    # Ensembling of the cross validation runs
    for j in metrics.keys():
        metrics[j] = np.mean(metrics[j], axis=0)
    prediction = np.mean(prediction, axis=0)
    label = "QM7_mean"
    plots(scores, prediction, label, save=True)
    print(metrics)

    # Show the gathered metrics
    show_figures(metrics, "QM7_evolution")


if __name__ == "__main__":
    main()
