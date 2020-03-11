import matplotlib.pyplot as plt
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras.optimizers import Adam

from CustomLayers import BiLSTMSelfAttentionLayer


def show_figures(metrics, label=None):
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
    plt.savefig('data/' + str(label) + '.png')
    print('Final MAE :', metrics["val_mae"][-1], 'Final MAPE :', metrics["val_mape"][-1])


def run_model(train_smiles, train_value, test_smiles, test_value, smiles, train_prot=None, test_prot=None, prot=None):
    """
    Runs the actual model on the specified data
    :param train_smiles: the embedded training smiles
    :param train_prot: the embedded training proteins
    :param train_value: the training value
    :param test_smiles: the embedded testing smiles
    :param test_prot: the embedded testing proteins
    :param test_value: the testing value
    :return: the results of the model fit as a History object
    """
    # General settings
    BATCH_SIZE = 64
    # Bi-LSTM Self-attention layer settings
    da = 15
    r = 10
    LSTM_SIZE = 10
    DROPOUT_RATE = 0
    # Training settings
    EPOCHS = 10

    input_smiles = Input(shape=(None, 100,), name="smiles")
    selfattention_smiles = BiLSTMSelfAttentionLayer(da, r, LSTM_SIZE, DROPOUT_RATE)(input_smiles)

    if prot is not None:
        input_protein = Input(shape=(None, 100,), name="protein")
        selfattention_protein = BiLSTMSelfAttentionLayer(da, r, LSTM_SIZE, DROPOUT_RATE)(input_protein)
        full = concatenate([selfattention_smiles, selfattention_protein])
    else:
        full = selfattention_smiles

    pred = Dense(1, activation="linear")(
        Dense(64, activation="linear")(Dense(128, activation="linear")(Dense(256, activation="relu")(full))))

    if prot is not None:
        model = Model(
            inputs=[input_smiles, input_protein],
            outputs=pred
        )
    else:
        model = Model(
            inputs=[input_smiles],
            outputs=pred
        )

    # utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True)
    opt = Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')

    model.compile(optimizer=opt,
                  loss="mse",
                  metrics=["mae", "mape"])
    if prot is not None:
        X = model.fit(x=[train_smiles, train_prot], y=train_value,
                      validation_data=([test_smiles, test_prot], test_value),
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS)
        pred = model.predict([smiles, prot])
    else:
        X = model.fit(x=[train_smiles], y=train_value,
                      validation_data=([test_smiles], test_value),
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS)
        pred = model.predict([smiles])

    return X, pred
