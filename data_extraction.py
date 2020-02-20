from Bio.PDB import PDBParser
from rdkit.Chem.PandasTools import LoadSDF


def get_SMILES_scores(sdf_filename):
    """
    Extract scores and SMILES of a sdf file
    sdf_filename = path of the file
    return 3 object :
    - SMILES : str
    - score : float
    - file : str, path of the file if it can't be read
    """
    df = LoadSDF(sdf_filename, smilesName='SMILES')
    file = ""
    try:
        score = max([float(x) for x in df["TOTAL"]])
        SMILES = [df["SMILES"]][0][0]
    except KeyError:
        SMILES = None
        score = None
        file += sdf_filename
    return SMILES, score, file


def get_residue(pdb_filename):
    """
    Return the list of residues of a pdb file
    pdb_filename : path of the file
    return the list of the index of the residues
    """
    sloppyparser = PDBParser(PERMISSIVE=True, QUIET=True)
    structure = sloppyparser.get_structure('MD_system', pdb_filename)
    res = []
    for model in structure:
        for chain in model:
            for residue in chain:
                res.append(residue.get_id()[1])
    return res


def convert_seq(seq_list):
    """
    Convert 3-letters code into 1-letter code
    seq_list : list of the 3-letters coded amino acids
    return a string made of the 1-letter code
    """
    seq = ""
    for i in seq_list:
        if i == "ALA":
            seq += "A"
        elif i == "ARG":
            seq += "R"
        elif i == "ASN":
            seq += "N"
        elif i == "ASP":
            seq += "D"
        elif i == "CYS":
            seq += "C"
        elif i == "GLN":
            seq += "Q"
        elif i == "GLU":
            seq += "E"
        elif i == "GLY":
            seq += "G"
        elif i == "HIS":
            seq += "H"
        elif i == "ILE":
            seq += "I"
        elif i == "LEU":
            seq += "L"
        elif i == "LYS":
            seq += "K"
        elif i == "MET":
            seq += "M"
        elif i == "PHE":
            seq += "F"
        elif i == "PRO":
            seq += "P"
        elif i == "SER":
            seq += "S"
        elif i == "THR":
            seq += "T"
        elif i == "TRP":
            seq += "W"
        elif i == "TYR":
            seq += "Y"
        elif i == "VAL":
            seq += "V"
        elif i == "SEC":
            seq += "U"
        elif i == "PYL":
            seq += "O"
        elif i == "HSE":
            seq += "S"
        elif i == "CYX":
            seq += "C"
        elif i == "HSD":
            seq += "H"
        else:
            print(i)
            seq += 'Test'
    return seq


def define_sequence(pdb_protein, pdb_filenames):
    """
    Return the sequence of interest of the protein composed of all interacting residues
    pdb_protein : file of the entire protein
    pdb_filenames : pdb files containing residues interacting with the compound
    return a string of the sequence of interest
    """
    residue_id = []
    for i in pdb_filenames:
        residue_id.append(get_residue(i))
    residue_id.sort()
    last = residue_id[-1]
    for i in range(len(residue_id) - 2, -1, -1):
        if last == residue_id[i]:
            del residue_id[i]
        else:
            last = residue_id[i]
    residue_id = residue_id[0]
    del residue_id[-1]
    sloppyparser = PDBParser(PERMISSIVE=True, QUIET=True)
    structure = sloppyparser.get_structure('MD_system', pdb_protein)
    residue = []
    model = structure.get_list()
    chain = model[0].get_list()
    for i in residue_id:
        resi = structure[0][chain[0].get_id()][i]
        residue.append(resi.get_resname())
    return convert_seq(residue)


def get_info(sdf_filenames, pdb_protein, pdb_filenames):
    """
    create a dictionnary of the SMILES, scores and the sequence of interest
    info returns the list of defective files
    """
    info = {"SMILES": [], "score": []}
    defect = []
    for i in sdf_filenames:
        smiles, score, file = get_SMILES_scores(i)
        info["SMILES"].append(smiles)
        info["score"].append(score)
        if file != "":
            defect.append(file)
    info["Sequence"] = define_sequence(pdb_protein, pdb_filenames)
    return defect, info


if __name__ == "__main__":
    sdf_test = ["data/Angel_dataset/scorp/fixed-conformers_3d_scorp (" + str(j + 1) + ").sdf" for j in range(
        71)] + ["data/Angel_dataset/scorp/fixed-conformers_3d_3d_scorp (" + str(i + 1) + ").sdf" for i in range(23)]
    pdb_test = ["data/Angel_dataset/Protein/bs_protein (" + str(i + 1) + ").pdb" for i in range(94)]
    print(get_info(sdf_test, "data/Angel_dataset/pro.pdb", pdb_test))
