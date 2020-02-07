import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Bidirectional, LSTM, Softmax
from tensorflow.keras import Model


class BiLSTMSelfAttentionModel(Model):
    """
    A model for a self-attention BiLSTM network, as described by Zhouhan Lin et al. in
     "A Structured Self-attentive Sentence Embedding".
     Consists of a bidirectional LSTM layer, outputting into a self-attention MLP,
      the results of which are combined with the output of the LSTM and then fed through a standard MLP with two layers.
    """
    def __init__(self, da, r, lstm_size):
        super(BiLSTMSelfAttentionModel, self).__init__()
        self.flatten = Flatten()
        self.biLSTM = Bidirectional(LSTM(lstm_size, return_sequences=True), merge_mode="concat")
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')
        self.da = da
        self.r = r
        self.attention1 = Dense(self.da, use_bias=False)
        self.attention2 = Dense(self.r, use_bias=False, activation="tanh")
        self.softmax = Softmax(axis=2)

    def self_attention(self, hidden):
        """
        The self-attention function, which performs executes the self-attention in the model.
        :param hidden: the output from the LSTM layer
        :return: the output from the self-attention function
        """
        mul1 = self.attention1(hidden)
        mul2 = self.attention2(mul1)
        return self.softmax(mul2)

    def call(self, x):
        x = self.biLSTM(x)
        y = self.self_attention(x)
        x = tf.matmul(y, x, transpose_a=True)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x


if __name__ == '__main__':
    model = BiLSTMSelfAttentionModel(15, 10, 10)

    batch_size = 64
    # Each MNIST image batch is a tensor of shape (batch_size, 28, 28).
    # Each input sequence will be of size (28, 28) (height is treated like time).
    input_dim = 28

    units = 64
    output_size = 10  # labels are from 0 to 9

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              batch_size=64,
              epochs=3)
