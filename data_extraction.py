from Bio.PDB import PDBParser
from rdkit.Chem.PandasTools import LoadSDF


def get_SMILES_scores(sdf_filename):
    df = LoadSDF(sdf_filename, smilesName='SMILES')
    file = []
    try:
        score = max([float(x) for x in df["TOTAL"]])
        SMILES = [df["SMILES"]][0][0]
    except KeyError:
        SMILES = None
        score = None
        file.append(sdf_filename)
    return SMILES, score, file


def get_residue(pdb_filename):
    sloppyparser = PDBParser(PERMISSIVE=True, QUIET=True)
    structure = sloppyparser.get_structure('MD_system', pdb_filename)
    res = []
    for model in structure:
        for chain in model:
            for residue in chain:
                res.append(residue.get_full_id())
    return res


def convert_seq(seq_list):
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
            seq+= 'Test'
    return seq


def define_sequence(pdb_protein, pdb_filenames):
    residue_id = []
    for i in pdb_filenames:
      residue_id += get_residue(i)
    sloppyparser = PDBParser(PERMISSIVE=True, QUIET=True)
    structure = sloppyparser.get_structure('MD_system', pdb_protein)
    seq = []
    for model in structure :
      for chain in model :
        for residue in chain :
          res = residue.get_full_id()
          if res in residue_id :
            seq.append(residue.get_resname())
    return convert_seq(seq)


def get_info(sdf_filenames, pdb_protein, pdb_filenames):
    info = {"SMILES": [], "score": []}
    defect = []
    for i in sdf_filenames:
        smiles, score, file = get_SMILES_scores(i)
        info["SMILES"].append(smiles)
        info["score"].append(score)
        if file != [] :
            defect.append(file)
    info["Sequence"] = define_sequence(pdb_protein, pdb_filenames)
    return defect, info


if __name__ == "__main__":
    sdf_test = ["/content/sample_data/fixed-conformers_3d_scorp (" + str(j + 1) + ").sdf" for j in range(
        82)]  + ["/content/sample_data/fixed-conformers_3d_3d_scorp (" + str(i+1) + ").sdf" for i in range(112)]
    pdb_test = ["/content/sample_data/bs_protein (" + str(i + 1) + ").pdb" for i in range(376)]
    print(get_info(sdf_test, "/content/sample_data/pro.pdb", pdb_test))
