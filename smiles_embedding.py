from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
from rdkit import Chem


def embed_smiles(smiles):
    model = word2vec.Word2Vec.load('data/model_300dim.pkl')
    mols = [Chem.MolFromSmiles(i) for i in smiles]
    sentences = sentences2vec([MolSentence(mol2alt_sentence(m, 1)) for m in mols], model, unseen='UNK')
    return sentences


if __name__ == '__main__':
    print(embed_smiles(["c1ccccc1", "c1ccccc1c2ccccc2"]))

