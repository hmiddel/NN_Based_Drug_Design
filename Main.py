import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Bidirectional, LSTM, Softmax, Layer, concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input, Model, utils
from CustomLayers import BiLSTMSelfAttentionLayer


if __name__ == '__main__':
    layer1 = BiLSTMSelfAttentionLayer(15, 10, 10)

    model = Sequential(
        [
            layer1,
            Flatten(),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ]
    )

    batch_size = 64
    # Each MNIST image batch is a tensor of shape (batch_size, 28, 28).
    # Each input sequence will be of size (28, 28) (height is treated like time).
    input_dim = 28

    units = 64
    output_size = 10  # labels are from 0 to 9

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    input_smiles = Input(shape=(None, 100, ), name="smiles")
    input_protein = Input(shape=(None, 100, ), name="protein")

    selfattention_smiles = BiLSTMSelfAttentionLayer(15, 10, 10)(input_smiles)
    selfattention_protein = BiLSTMSelfAttentionLayer(15, 10, 10)(input_protein)

    full = concatenate([selfattention_smiles, selfattention_protein])

    pred = Dense(1)(Dense(20, activation="tanh")(full))

    model = Model(
        inputs=[input_smiles, input_protein],
        outputs=pred
    )

    utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True)

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              batch_size=64,
              epochs=3)
