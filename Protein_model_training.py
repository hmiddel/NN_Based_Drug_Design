import re

from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile


def protein_train(dims, data, n, window_size, negative_size):
    """
    word2vec method to embed protein sequence, based on
     https://github.com/dqwei-lab/SPVec/blob/master/examples/Biochemical%20implications%20of%20ProtVec%20features.ipynb
    :param dims: length of the output vectors
    :param data: input data in the form of an iterable containing protein sequences
    :param n: the k-mer size to use
    :param window_size: maximum distance between the current and predicted word within a sentence
    :param negative_size: If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
            should be drawn (usually between 5-20).
    :save the model in a file
    """
    texts = []
    prot_words = [[] for _ in range(len(list(data)))]
    get_tmpfile("protein_embedding.model")
    for (j, document) in enumerate(list(data)):
        prot_words[j] = [word for word in re.findall(r'.{' + str(n) + '}', document)]
        texts += [prot_words[j]]
    model = Word2Vec(texts, size=dims, window=window_size, min_count=1, negative=negative_size, sg=1, sample=0.001,
                     hs=1, workers=2)
    model.wv.save("data/protein_embedding.model")

if __name__ == "__main__" :
    data = set()
    with open("data/BindingDB_All.tsv", encoding='utf-8') as i :
        for line in i :
            splitline = line.split("\t")
            data.add(splitline[37])
    protein_train(100, data, 3, 5, 5)