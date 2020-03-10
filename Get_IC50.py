from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, MolSentence, sentences2vec
from rdkit import Chem

from Protein_embedding import embed_protein


def get_IC50():
    """
    Write a file containing the IC50, SMILES, SMILES embedding and protein embedding from the BindingDB dataset
    Input file size is 3,5Gb
    Output file size is around 25Gb
    """
    # Get all protein sequences
    Protein = []
    with open('data/BindingDB_All.tsv', encoding='utf-8') as i:
        for line in i:
            splitline = line.split("\t")
            Protein.append(splitline[37])
    # Delete the header
    del Protein[0]
    # Embed the sequences
    protein_embed = embed_protein(100, Protein, 3, 5, 5)

    model = word2vec.Word2Vec.load('data/model_300dim.pkl')
    with open('data/BindingDB_All.tsv', encoding='utf-8') as i:
        with open('data/BindingDB_IC50.tsv', 'w') as o:
            for z, line in enumerate(i):
                splitline = line.split("\t")

                # Write the header
                if z == 0:
                    o.write(
                        "IC50" + "\t" + "Ligand SMILES" + "\t" + "SMILES embedding" + "\t" + "Protein embedding" + "\n")

                # Write the info only when the IC50 and the SMILES code are valid
                else:
                    if splitline[9] != ("" and 0):
                        if ("<" not in splitline[9]) and (">" not in splitline[9]):
                            try:
                                m = Chem.MolFromSmiles(splitline[1])
                                smiles_embedding = sentences2vec(MolSentence(mol2alt_sentence(m, 1)), model,
                                                                 unseen='UNK')
                                o.write(str(splitline[9]) + "\t" + str(splitline[1]) + "\t" + str(
                                    smiles_embedding.tolist()) + "\t" + str(next(protein_embed)) + "\n")
                            except TypeError:
                                next(protein_embed)


if __name__ == "__main__":
    get_IC50()
