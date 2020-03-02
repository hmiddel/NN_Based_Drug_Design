import re

from gensim.models import KeyedVectors


def embed_protein(data, n):
    """
    word2vec method to embed protein sequence, based on
     https://github.com/dqwei-lab/SPVec/blob/master/examples/Biochemical%20implications%20of%20ProtVec%20features.ipynb
    :param data: input data in the form of an iterable containing protein sequences
    :param n: the k-mer size to use
    :return: the embedded vector for each k-mer in the protein sequence
    """
    texts = []
    prot_words = [[] for _ in range(len(list(data)))]
    for (j, document) in enumerate(list(data)):
        prot_words[j] = [word for word in re.findall(r'.{' + str(n) + '}', document)]
        texts += [prot_words[j]]
    model = KeyedVectors.load("data/protein_embedding.model", mmap='r')
    vectors = ([list(model[word]) for word in (prot_words[i])] for i in range(len(prot_words)))
    return vectors
