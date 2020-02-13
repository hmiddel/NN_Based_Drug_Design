import pandas as pd
import re
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def embed_protein(dims, data, n, window_size, negative_size):
    """
    word2vec method to embed protein sequence, based on
     https://github.com/dqwei-lab/SPVec/blob/master/examples/Biochemical%20implications%20of%20ProtVec%20features.ipynb
    :param dims: length of the output vectors
    :param data: input data in the form of an iterable containing protein sequences
    :param n: the k-mer size to use
    :param window_size: maximum distance between the current and predicted word within a sentence
    :param negative_size: If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
            should be drawn (usually between 5-20).
    :return: the embedded vector for each k-mer in the protein sequence
    """
    texts = []
    prot_words = [[] for _ in range(len(list(data)))]
    for (j, document) in enumerate(list(data)):
        prot_words[j] = [word for word in re.findall(r'.{' + str(n) + '}', document)]
        texts += [prot_words[j]]
    model = Word2Vec(texts, size=dims, window=window_size, min_count=1, negative=negative_size, sg=1, sample=0.001,
                     hs=1, workers=4)
    vectors = [[model[word] for word in (prot_words[i])] for i in range(len(prot_words))]
    return vectors


def object_mean(embedding):
    """
    Embedding of each protein in a vector of length dim which is the mean of all of its k-mers
    :param embedding: the word embedding to convert to a protein embedding
    :return: a list containing protein embeddings
    """
    for i in range(len(embedding)):
        embedding[i] = np.mean(embedding[i], axis=0)
    return embedding


if __name__ == '__main__':
    a = [x for x in range(0, 1033, 2)]  # lines of names
    b = [x for x in range(1, 1032, 2)]  # lines of sequences

    # Creating a DataFrame of the name, family and sequences of human kinases
    prot_seq = pd.read_csv('data/human_protein_sequences.fasta', sep=' ', skiprows=a, header=None)
    prot_seq.columns = ['Sequence']
    prot_name = pd.read_csv('data/human_protein_sequences.fasta', sep=' ', skiprows=b, header=None)
    prot_name.columns = ['Name', 'Family']
    kinome_data = prot_name.join(prot_seq, how='right')
    kinome_data.head()

    # embedding of each k-mers in a vector of length dim using word2vec
    dim = 100
    k = 3
    kinome_seq = kinome_data['Sequence']
    prot_vec = embed_protein(dim, kinome_seq, k, 5, 5)

    embedded_data = pd.DataFrame(object_mean(prot_vec))
    embedded_data_labeled = kinome_data.join(embedded_data, how='right')  # merging info and embedding
    embedded_data_labeled.dropna(axis=1, how='all')

    # Dimensionality reduction to plot the protein embeddings into a 2D graph
    tsne = TSNE(n_components=2)
    X = tsne.fit_transform(embedded_data_labeled.iloc[:, 4:])

    # Plotting the results
    color_1 = embedded_data.iloc[:, 1]
    figure = plt.figure()
    figure.suptitle('Repartition of kinase families after embedding', fontsize=16)
    figure = plt.scatter(X[:, 0], X[:, 1], c=color_1, marker='.', cmap=plt.cm.rainbow)
    plt.show()

if __name__ == "__main__":
    data = pd.read_csv("data/binding_data_cleared2.tsv", sep="\t")
    seq = data['BindingDB Target Chain  Sequence']
    seq = seq[:50]
    embed = embed_protein(100, seq, 3, 5, 5)
    print(len(embed), len(embed[5]), len(embed[5][0]))
