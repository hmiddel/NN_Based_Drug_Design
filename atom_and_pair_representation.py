import rdkit.Chem as ch
import rdkit.Chem.AllChem as AllChem

ATOM_TYPES = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
METALS = []
METALS.extend(range(3, 5))
METALS.extend(range(11, 14))
METALS.extend(range(19, 32))
METALS.extend(range(37, 51))
METALS.extend(range(55, 85))
METALS.extend(range(87, 119))

# These are the definitions RDKit uses for hydrogen bond donors and acceptors
HDonorSmarts = ch.MolFromSmarts('[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]')

HAcceptorSmarts = ch.MolFromSmarts('[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),' +
                                   '$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),' +
                                   '$([nH0,o,s;+0])]')


def get_one_hot_atom_type(atom):
    """
    Generates a one-hot encoding of the element of the atom, based on the following list:
    H, C, N, O, F, P, S, Cl, Br, I or metal
    :param atom: the atom to generate the encoding for
    :return: a one-hot encoding of the element of the atom
    """
    encoding = [0 for _ in range(11)]
    try:
        index = ATOM_TYPES.index(atom.GetAtomicNum())
        encoding[index] = 1
        return encoding
    except ValueError:
        if atom.GetAtomicNum() in METALS:
            encoding[10] = 1
        return encoding


def get_one_hot_chirality(atom):
    """
    Generates a one-hot encoding for the chirality of the atom
    :param atom: the atom to generate the encoding for
    :return: the generated encoding
    """
    encoding = [0, 0]
    if atom.GetChiralTag() == ch.CHI_TETRAHEDRAL_CW:
        encoding[0] = 1
    elif atom.GetChiralTag() == ch.CHI_TETRAHEDRAL_CCW:
        encoding[1] = 1
    return encoding


def get_ring_sizes(molecule, atom):
    """
    Generates a list of ring size counts,
     which are the amount of times the specified atom is contained in a ring of a certain size
    :param molecule: the molecule this atom comes from, to get the rings
    :param atom: the atom for which the ring size counts need to be determined
    :return: a list of ring size counts, starting at ring size 3 and ending at size 8
    """
    sizes = [0 for _ in range(3, 9)]
    for ring in ch.GetSymmSSSR(molecule):
        if atom.GetIdx() in ring:
            sizes[len(ring) - 3] += 1
    return sizes


def get_one_hot_hybridization(atom):
    """
    Generates a one-hot encoding for the hybridization state of the atom, of the form [SP1, SP2, SP3] or none if it is none of those.
    :param atom: the atom to generate the encoding for
    :return: the generated encoding
    """
    encoding = [0 for _ in range(3)]
    hybridization = atom.GetHybridization()
    if str(hybridization) == "SP1":
        encoding[0] = 1
    elif str(hybridization) == "SP2":
        encoding[1] = 1
    elif str(hybridization) == "SP3":
        encoding[2] = 1
    return encoding


def get_hydrogen_bonding(molecule, atom):
    hbonds = [0, 0]
    if molecule.HasSubstructMatch(HDonorSmarts):
        if atom.GetIdx() in [idx for match in molecule.GetSubstructMatches(HDonorSmarts) for idx in match]:
            hbonds[0] = 1
    if molecule.HasSubstructMatch(HAcceptorSmarts):
        if atom.GetIdx() in [idx for match in molecule.GetSubstructMatches(HAcceptorSmarts) for idx in match]:
            hbonds[1] = 1
    return hbonds


def get_atom_properties(molecule):
    """
    Gets a list of properties for each atom, being the following:
    A one-hot encoding of the atom type
    The formal charge
    The Gasteiger partial charge
    A one-hot encoding of the chirality
    For each ring size (3–8), the number of rings that include this atom
    A one-hot encoding of the hybridization state
    A binary encoding of hydrogen donator/acceptor function
    A boolean aromaticity flag
    :param molecule: the molecule for which to determine these properties for each atom
    :return: a list of the calculated properties
    """
    properties = []
    AllChem.ComputeGasteigerCharges(molecule)
    for atom in molecule.GetAtoms():
        atom_props = []
        atom_props.extend(get_one_hot_atom_type(atom))
        atom_props.append(atom.GetFormalCharge())
        atom_props.append(atom.GetPropsAsDict()["_GasteigerCharge"])
        atom_props.extend(get_one_hot_chirality(atom))
        atom_props.extend(get_ring_sizes(molecule, atom))
        atom_props.extend(get_one_hot_hybridization(atom))
        atom_props.extend(get_hydrogen_bonding(molecule, atom))
        atom_props.append(int(atom.GetIsAromatic()))
        properties.append(atom_props)
    return properties


if __name__ == '__main__':
    molecule = ch.MolFromSmiles(
        "c1ccccc1O")
    print(get_atom_properties(molecule))
