import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense

from CustomLayers import BiLSTMSelfAttentionLayer
from smiles_embedding import embed_smiles
from training_plot import plots

# General settings
BATCH_SIZE = 64
# Dataset settings
DATASET_FRACTION = 1
CROSS_VALIDATION_NUMBER = 5
# Bi-LSTM Self-attention layer settings
da = 15
r = 10
LSTM_SIZE = 100
DROPOUT_RATE = 0.1
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
    plt.savefig('data/evolution_qm7.png')
    print('Final MAE :', metrics["val_mae"][-1], 'Final MAPE :', metrics["val_mape"][-1])


def run_model_smiles(train_smiles, train_IC, test_smiles, test_IC, smiles):
    """
    Runs the actual model on the specified data
    :param train_smiles: the embedded training smiles
    :param train_IC: the training IC50
    :param test_smiles: the embedded testing smiles
    :param test_IC: the testing IC50
    :return: the results of the model fit as a History object
    """
    input_smiles = Input(shape=(None, 100,), name="smiles")

    selfattention_smiles = BiLSTMSelfAttentionLayer(da, r, LSTM_SIZE, DROPOUT_RATE)(input_smiles)

    full = selfattention_smiles

    pred = Dense(1, activation="linear")(
        Dense(64, activation="linear")(Dense(128, activation="linear")(Dense(256, activation="relu")(full))))

    model = Model(
        inputs=[input_smiles],
        outputs=pred
    )

    # utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True)

    model.compile(optimizer="adam",
                  loss="mse",
                  metrics=["mae", "mape"])

    X = model.fit(x=[train_smiles], y=train_IC,
                  validation_data=([test_smiles], test_IC),
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS)

    pred = model.predict(smiles)
    return X, pred


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
        hist, pred = run_model_smiles(embedded_train_smiles, train_IC, embedded_test_smiles, test_IC, smiles)
        X.append(hist)
        prediction.append(pred)

    # Gather the metrics and plot prediction comparison for each cross validation run
    metrics = {'loss': [], 'mae': [], 'mape': [], 'val_loss': [], 'val_mae': [], 'val_mape': []}
    for i in range(CROSS_VALIDATION_NUMBER):
        for j in metrics.keys():
            metrics[j].append(X[i].history[j])
        label = 'qm7_' + str(i)
        plots(scores, prediction[i], label, save=True)

    # Ensembling of the cross validation runs
    for j in metrics.keys():
        metrics[j] = np.mean(metrics[j], axis=0)
    prediction = np.mean(prediction, axis=0)
    label = "qm7_mean"
    plots(scores, prediction, label, save=True)
    print(metrics)

    # Show the gathered metrics
    show_figures(metrics)


if __name__ == "__main__":
    main()