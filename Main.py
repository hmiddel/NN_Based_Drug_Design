import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Bidirectional, LSTM, Softmax
from tensorflow.keras import Model



class FullModel(Model):
    def __init__(self):
        super(FullModel, self).__init__()
        self.flatten = Flatten()
        self.biLSTM = Bidirectional(LSTM(10, return_sequences=True), merge_mode="concat")
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')
        self.da = 15
        self.r = 10
        self.attention1 = Dense(self.da, use_bias=False)
        self.attention2 = Dense(self.r, use_bias=False, activation="tanh")
        self.softmax = Softmax(axis=2)


    def self_attention(self, hidden):
        mul1 = self.attention1(hidden)
        # mul1 = (Batch, 28, 15)
        mul2 = self.attention2(mul1)
        # mul2 = (Batch, 28, 10)
        return self.softmax(mul2)

    def call(self, x):
        # shape of x here: (Batch, 28, 28)
        x = self.biLSTM(x)
        # shape of x here: (Batch, 28, 20)
        y = self.self_attention(x)
        # shape of y here: (Batch, 28, 10)
        x = tf.matmul(y, x, transpose_a=True)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x


if __name__ == '__main__':
    model = FullModel()

    batch_size = 64
    # Each MNIST image batch is a tensor of shape (batch_size, 28, 28).
    # Each input sequence will be of size (28, 28) (height is treated like time).
    input_dim = 28

    units = 64
    output_size = 10  # labels are from 0 to 9

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    sample, sample_label = x_train[0], y_train[0]

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              batch_size=64,
              epochs=3)
