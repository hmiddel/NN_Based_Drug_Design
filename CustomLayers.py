import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, Bidirectional, LSTM, Softmax, Layer


class BiLSTMSelfAttentionLayer(Layer):
    """
    A model for a self-attention BiLSTM network, as described by Zhouhan Lin et al. in
     "A Structured Self-attentive Sentence Embedding".
     Consists of a bidirectional LSTM layer, outputting into a self-attention MLP,
      the results of which are combined with the output of the LSTM and then fed through a standard MLP with two layers.
    """
    def __init__(self, da, r, lstm_size, dropout_rate):
        super(BiLSTMSelfAttentionLayer, self).__init__()
        self.flatten = Flatten()
        self.dropout = Dropout(dropout_rate)
        self.biLSTM = Bidirectional(LSTM(lstm_size, activation='relu', return_sequences=True), merge_mode="concat")
        self.da = da
        self.r = r
        self.attention1 = Dense(self.da, use_bias=False)
        self.attention2 = Dense(self.r, use_bias=False, activation="relu")
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
        x = self.dropout(x)
        y = self.self_attention(x)
        x = tf.matmul(y, x, transpose_a=True)
        x = tf.reduce_sum(x, 2)
        return x
