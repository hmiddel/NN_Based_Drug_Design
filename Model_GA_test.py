import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from data_extraction import get_all_scores
from smiles_embedding import embed_smiles
from training_plot import plots

# Dataset settings
DATASET_FRACTION = 1
CROSS_VALIDATION_NUMBER = 5


def main():
    # Load first dataset, produced by Angel
    sdf_test = ["data/GA/Angel_dataset/scorp/fixed-conformers_3d_scorp (" + str(j + 1) + ").sdf" for j in range(71)] + [
        "data/GA/Angel_dataset/scorp/fixed-conformers_3d_3d_scorp (" + str(i + 1) + ").sdf" for i in range(23)]
    _, sdf_test = get_all_scores(sdf_test)
    data_angel = pd.DataFrame(sdf_test)
    data_angel = data_angel.dropna()
    data_angel.to_csv("data/GA/Angel_dataset/scores.csv", index=False)

    # Load second dataset, produced by Xin
    sdf_test = ["data/GA/Xin_dataset/scorp/fixed-conformers_3d_scorp Xin(" + str(j + 1) + ").sdf" for j in
                range(71)] + ["data/GA/Xin_dataset/scorp/fixed-conformers_3d_3d_scorp Xin(" + str(i + 1) + ").sdf" for i
                              in range(23)]
    _, sdf_test = get_all_scores(sdf_test)
    data_xin = pd.DataFrame(sdf_test)
    data_xin = data_xin.dropna()
    data_angel.to_csv("data/GA/Xin_dataset/scores.csv", index=False)

    # Load third dataset, produced by Shabnam
    sdf_test = ["data/GA/Shabnam_dataset/scorp/fixed-conformers_3d_scorp (" + str(j + 1) + ").sdf" for j in
                range(82)] + ["data/GA/Shabnam_dataset/scorp/fixed-conformers_3d_3d_scorp (" + str(i + 1) + ").sdf" for
                              i in range(112)]
    _, sdf_test = get_all_scores(sdf_test)
    data_Shabnam = pd.DataFrame(sdf_test)
    data_Shabnam = data_Shabnam.dropna()
    data_angel.to_csv("data/GA/Shabnam_dataset/scores.csv", index=False)

    """
    # Combine datasets
    data = pd.concat([data_angel, data_xin], axis=0)
    data = data.sample(frac=DATASET_FRACTION)
    data = np.array_split(data, CROSS_VALIDATION_NUMBER)
    """
    # Run the model multiple times for cross validation
    for data, name in [(data_xin, "Xin"), (data_Shabnam, "Shabnam"), (data_angel, "Angel")]:
        X = []
        prediction = []
        mean = np.mean(data["score"])
        sd = np.std(data["score"])
        data['score'] = (data['score'] - mean) / sd
        scores = list(data['score'])

        data["SMILES"] = embed_smiles(data["SMILES"])

        data = np.array_split(data, CROSS_VALIDATION_NUMBER)

        for i in range(CROSS_VALIDATION_NUMBER):
            test_data = pd.DataFrame(data[i])
            dataset = data[:]
            del dataset[i]
            train_data = pd.concat(dataset)
            del dataset

            # Load the data and normalize it, if needed
            train_smiles, test_smiles = train_data['SMILES'], test_data['SMILES']
            embedded_train_smiles = tf.ragged.constant(train_smiles).to_tensor(shape=(None, None, 100))
            embedded_test_smiles = tf.ragged.constant(test_smiles).to_tensor(shape=(None, None, 100))

            smiles = train_smiles + test_smiles
            smiles = tf.ragged.constant(smiles).to_tensor(shape=(None, None, 100))

            train_IC, test_IC = np.array(train_data["score"]), np.array(test_data["score"])
            del train_data, test_data

            # Run the model
            hist, pred = run_model(embedded_train_smiles, train_IC, embedded_test_smiles, test_IC, smiles)
            X.append(hist)
            prediction.append(pred)

        # Gather the metrics for each cross validation run
        metrics = {'loss': [], 'mae': [], 'mape': [], 'val_loss': [], 'val_mae': [], 'val_mape': []}
        for i in range(CROSS_VALIDATION_NUMBER):
            for j in metrics.keys():
                metrics[j].append(X[i].history[j])
            label = "GA_" + str(name) + str(i)
            plots(scores, prediction[i], label=label, save=False)
        for j in metrics.keys():
            metrics[j] = np.mean(metrics[j], axis=0)
        prediction = np.mean(prediction, axis=0)
        label = "GA_" + str(name) + "_mean"
        plots(scores, prediction, label, save=True)
        print(metrics)
        show_figures(metrics, "GA_"+str(name))


if __name__ == '__main__':
    main()
