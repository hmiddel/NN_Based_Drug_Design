from tensorflow import keras

import pandas as pd
import tensorflow as tf

from smiles_embedding import embed_single_smiles
from Protein_embedding import embed_protein

def prediction(SMILES,prot_names,prot_seq):
    """
    SMILES: list of SMILES
    protein_info: list of list [[protein_name,...,protein_name],[protein_sequence,...,protein_sequence]]
    """
    embedded_smiles = []
    smiles= []
    for j, mol in enumerate(SMILES):
        try:
            embedded_smiles.append(embed_single_smiles(mol))
            smiles.append(mol)
        except TypeError:
            pass
    print("smiles embedded")
    embedded_prot = embed_protein(prot_seq)
    print("protein embedded")
    model = tf.keras.models.load_model('data/model_save')
    print("model loaded")
    prediction = []
    for j, prot in enumerate(embedded_prot):
        prot = tf.ragged.constant([prot]).to_tensor(shape=(1, None, 100))
        for i, compound in enumerate(embedded_smiles):
            compound = tf.ragged.constant([compound]).to_tensor(shape=(1, None, 100))
            prediction.append([smiles[i], 10**float(model.predict(x=[compound, prot]))])
            prediction.sort(key=lambda x: x[1])
    return prediction


if __name__=="__main__":
    data = pd.read_csv("data/test.tsv", sep="\t")
    smiles = data["Ligand SMILES"]
    prot_names = ['Integrase']
    prot_seq = ["MGARASVLSGGELDRWEKIRLRPGGKKKYKLKHIVWASRELERFAVNPGLLETSEGCRQILGQLQPSLQTGSEELRSLYNTVATLYCVHQRIEIKDTKEALDKIEEEQNKSKKKAQQAAADTGHSNQVSQNYPIVQNIQGQMVHQAISPRTLNAWVKVVEEKAFSPEVIPMFSALSEGATPQDLNTMLNTVGGHQAAMQMLKETINEEAAEWDRVHPVHAGPIAPGQMREPRGSDIAGTTSTLQEQIGWMTNNPPIPVGEIYKRWIILGLNKIVRMYSPTSILDIRQGPKEPFRDYVDRFYKTLRAEQASQEVKNWMTETLLVQNANPDCKTILKALGPAATLEEMMTACQGVGGPGHKARVLAEAMSQVTNSATIMMQRGNFRNQRKIVKCFNCGKEGHTARNCRAPRKKGCWKCGKEGHQMKDCTERQANFLREDLAFLQGKAREFSSEQTRANSPTRRELQVWGRDNNSPSEAGADRQGTVSFNFPQVTLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNFPISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPVFAIKKKDSTKWRKLVDFRELNKRTQDFWEVQLGIPHPAGLKKKKSVTVLDVGDAYFSVPLDEDFRKYTAFTIPSINNETPGIRYQYNVLPQGWKGSPAIFQSSMTKILEPFRKQNPDIVIYQYMDDLYVGSDLEIGQHRTKIEELRQHLLRWGLTTPDKKHQKEPPFLWMGYELHPDKWTVQPIVLPEKDSWTVNDIQKLVGKLNWASQIYPGIKVRQLCKLLRGTKALTEVIPLTEEAELELAENREILKEPVHGVYYDPSKDLIAEIQKQGQGQWTYQIYQEPFKNLKTGKYARMRGAHTNDVKQLTEAVQKITTESIVIWGKTPKFKLPIQKETWETWWTEYWQATWIPEWEFVNTPPLVKLWYQLEKEPIVGAETFYVDGAANRETKLGKAGYVTNRGRQKVVTLTDTTNQKTELQAIYLALQDSGLEVNIVTDSQYALGIIQAQPDQSESELVNQIIEQLIKKEKVYLAWVPAHKGIGGNEQVDKLVSAGIRKVLFLDGIDKAQDEHEKYHSNWRAMASDFNLPPVVAKEIVASCDKCQLKGEAMHGQVDCSPGIWQLDCTHLEGKVILVAVHVASGYIEAEVIPAETGQETAYFLLKLAGRWPVKTIHTDNGSNFTGATVRAACWWAGIKQEFGIPYNPQSQGVVESMNKELKKIIGQVRDQAEHLKTAVQMAVFIHNFKRKGGIGGYSAGERIVDIIATDIQTKELQKQITKIQNFRVYYRDSRNPLWKGPAKLLWKGEGAVVIQDNSDIKVVPRRKAKIIRDYGKQMAGDDCVASRQDED"]
    pred = prediction(smiles,prot_names,prot_seq)
    output = pd.DataFrame(pred)
    output.columns = ["Ligand SMILES", "predicted IC50"]
    print(output.head())
    output = output.merge(data[["IC50 (nM)","Ligand SMILES"]], how='right', on="Ligand SMILES")
    output = output[["predicted IC50","IC50 (nM)", "Ligand SMILES"]]
    output.to_csv("data/predictions.tsv", sep="\t")