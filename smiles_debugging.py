import pandas as pd
import smiles_embedding

if __name__ == '__main__':
    data = pd.read_csv(
        "D:\\Corentin\\Documents\\Stage_M1\\Deep_learning_models\\NN_Based_Drug_Design\\data\\binding_data_dropna.tsv",
        sep='\t')
    smiles = data['Ligand SMILES']
    smiles_embed = smiles_embedding.embed_smiles(smiles)