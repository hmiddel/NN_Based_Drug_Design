from Bio.PDB import PDBParser
from rdkit.Chem.PandasTools import LoadSDF

AA_CODES = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O", "HSE": "S", "CYX": "C",
    "HSD": "H"
}


def get_SMILES_scores(sdf_filename):
    """
    Gets smiles and the highest scorpion score from a scorpion sdf file
    :param sdf_filename: the sdf file to read from
    :return: the found smiles, the found score, and the file if an error occured, or None otherwise
    """
    df = LoadSDF(sdf_filename, smilesName='SMILES')
    file = None
    try:
        score = max([float(x) for x in df["TOTAL"]])
        SMILES = [df["SMILES"]][0][0]
    except KeyError:
        SMILES = None
        score = None
        file = sdf_filename
    return SMILES, score, file


def get_all_scores(sdf_filenames):
    """
    Gets all the smiles and scores from the sdf files
    :param sdf_filenames: a list of sdf files to parse and extract smiles and scores from
    :return: a dictionary containing a list of smiles and a list of scores
    """
    info = {"SMILES": [], "score": []}
    defect = []
    for file in sdf_filenames:
        sm, sc, filename = get_SMILES_scores(file)
        info["SMILES"].append(sm)
        info["score"].append(sc)
        if file:
            defect.append(filename)
    return defect, info


def get_residue(pdb_filename):
    """
    Gets a list of residues in the passed pdb files
    :param pdb_filename: the pdb file
    :return: a list of residues as full residue ids as produced by biopython
    """
    sloppyparser = PDBParser(PERMISSIVE=True, QUIET=True)
    structure = sloppyparser.get_structure('MD_system', pdb_filename)
    res = []
    for model in structure:
        for chain in model:
            for residue in chain:
                res.append(residue.get_full_id())
    return res


def convert_seq(seq_list):
    """
    Converts a sequence of three letter amino acid ids into a list of single letter amino acid ids.
    Inserts Test into the list for unknown amino acids.
    :param seq_list: a list of three letter amino acid ids
    :return: a list of single letter amino acid ids
    """
    seq = ""
    for i in seq_list:
        if i in AA_CODES.keys():
            seq += AA_CODES[i]
        else:
            print(i)
            seq += 'Test'
    return seq


def define_sequence(pdb_protein, pdb_filenames):
    """
    Defines the sequence of all residues that appear at least once in a list of pdb files
    :param pdb_protein: the reference protein, used to decode ids into residue names
    :param pdb_filenames: a list of protein files that are used to define the sequence
    :return: a list of single letter amino acid ids that appear at least once in the given list of pdb files
    """
    residue_id = []
    for i in pdb_filenames:
        residue_id += get_residue(i)
    sloppyparser = PDBParser(PERMISSIVE=True, QUIET=True)
    structure = sloppyparser.get_structure('MD_system', pdb_protein)
    seq = []
    for model in structure:
        for chain in model:
            for residue in chain:
                res = residue.get_full_id()
                if res in residue_id:
                    seq.append(residue.get_resname())
    return convert_seq(seq)


def get_info(sdf_filenames, pdb_protein, pdb_filenames):
    """
    Gets a dictionary of smiles, scores, and protein sequences, one for each sdf file passed
    :param sdf_filenames: a list of sdf files to parse and extract smiles and scores from.
    :param pdb_protein: a reference protein
    :param pdb_filenames: a list of proteins containing only residues in the pocket used for scoring
    :return: a dictionary containing a list of smiles, a list of scores and a list of protein sequences
    """
    defect, info = get_all_scores(sdf_filenames)
    info["sequence"] = define_sequence(pdb_protein, pdb_filenames)
    return defect, info


if __name__ == "__main__":
    sdf_test = ["data/Shabnam_dataset/scorp/fixed-conformers_3d_scorp (" + str(j + 1) + ").sdf" for j in range(
        82)] + ["data/Shabnam_dataset/scorp/fixed-conformers_3d_3d_scorp (" + str(i + 1) + ").sdf" for i in range(112)]
    pdb_test = ["data/Shabnam_dataset/prot/bs_protein (" + str(i + 1) + ").pdb" for i in range(376)]
    print(get_info(sdf_test, "data/Shabnam_dataset/pro.pdb", pdb_test))
