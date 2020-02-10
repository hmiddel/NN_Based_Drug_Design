from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
from rdkit import Chem
import numpy as np

def embed_smiles(smiles):
<<<<<<< HEAD
    model = word2vec.Word2Vec.load('D:/Corentin/Documents/Stage_M1/Deep_learning_models/NN_Based_Drug_Design/data/model_300dim.pkl')
    mols = (Chem.MolFromSmiles(i) for i in smiles)
    sentences = [sentences2vec(MolSentence(mol2alt_sentence(m, 1)), model, unseen='UNK') for m in mols]
    vecs = [DfVec(x).vec for x in sentences]
    return vecs
=======
    model = word2vec.Word2Vec.load('data/model_300dim.pkl')
    mols = (Chem.MolFromSmiles(i) for i in smiles)
    sentences = sentences2vec((MolSentence(mol2alt_sentence(m, 1)) for m in mols), model, unseen='UNK')
    return sentences
>>>>>>> 0c529ee3ec07d8b869b62569ab9b05f59a4c2351


if __name__ == '__main__':
        smiles_embedded= embed_smiles(["c1ccccc1","O=C1CCCC2=C1C1(CCS(=O)(=O)C1)N=C(Nc1nc3ccccc3o1)N2","CN[C@@H]1C[C@H]2O[C@@](C)([C@@H]1OC)n1c3ccccc3c3c4CNC(=O)c4c4c5ccccc5n2c4c13"])