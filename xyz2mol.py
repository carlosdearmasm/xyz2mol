#
# Written by Jan H. Jensen based on this paper Yeonjoon Kim and Woo Youn Kim 
# "Universal Structure Conversion Method for Organic Molecules:
# From Atomic Connectivity to Three-Dimensional Geometry" Bull. Korean Chem.
# Soc. 2015, Vol. 36, 1769-1777 DOI: 10.1002/bkcs.10334
# 
# modified by Joaquim Jornet and Carlos de Armas
# date July 10th, 2018
#
import copy
import itertools
import numpy as np
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import AllChem


def get_ua(max_valence_list, valence_list):
    ua = []
    du = []
    for i, (maxValence, valence) in enumerate(zip(max_valence_list,
                                                  valence_list)):
        if maxValence - valence > 0:
            ua.append(i)
            du.append(maxValence - valence)
    return ua, du


def get_atoms_min_connectivity(ac, atomic_num_list):
    heavy_atom_connectivity = []
    for atomic_number, row in zip(atomic_num_list, ac):
        count = 0
        for bond, atomic_number2 in zip(row, atomic_num_list):
            if atomic_number2 != 1 and bond == 1:
                count += 1
        if atomic_number == 1:
            # no multiple bonds to H's, so ignore
            count = 9
        heavy_atom_connectivity.append(count)

    min_connectivity = min(heavy_atom_connectivity)
    atoms_min_connectivity = []
    for i, connectivity in enumerate(heavy_atom_connectivity):
        if connectivity == min_connectivity:
            atoms_min_connectivity.append(i)
    return atoms_min_connectivity


def get_bo_old(ac, valences):
    bo = ac.copy()
    bo_valence = list(bo.sum(axis=1))
    ua, du = get_ua(valences, bo_valence)
    # construct the bond order but not in a random way, it use the same
    # order read it from the xyz
    while len(du) > 1:
        ua_pairs = itertools.combinations(ua, 2)
        for i, j in ua_pairs:
            if bo[i, j] > 0:
                bo[i, j] += 1
                bo[j, i] += 1
                break
        bo_valence = list(bo.sum(axis=1))
        ua_new, du_new = get_ua(valences, bo_valence)
        if du_new != du:
            ua = copy.copy(ua_new)
            du = copy.copy(du_new)
        else:
            break
    return bo


def get_bo(ac, p_ua, du, valences):
    bo = ac.copy()
    ua = list(p_ua)
    while len(du) > 1:
        ua_pairs = itertools.combinations(ua, 2)
        for i, j in ua_pairs:
            if bo[i, j] > 0:
                bo[i, j] += 1
                bo[j, i] += 1
                break
        bo_valence = list(bo.sum(axis=1))
        # bo_valence = list(bo.sum(axis=1)[np.array(ua)])
        ua_new, du_new = get_ua(valences, bo_valence)
        if du_new != du:
            # ua = copy.copy(ua_new)
            ua = [ua_i for ua_i in p_ua if ua_i in ua_new]
            du = copy.copy(du_new)
        else:
            break
    return bo


def bo_is_ok(bo, ac, charg, du,
             atomic_valence_electrons,
             atomic_num_list,
             charged_frag):
    q = 0
    if charged_frag:
        bo_valences = list(bo.sum(axis=1))
        for i, atom in enumerate(atomic_num_list):
            q += get_atomic_charge(atom,
                                   atomic_valence_electrons[atom],
                                   bo_valences[i])
            if atom == 6:
                number_of_single_bonds_to_c = \
                    list(bo[i, :]).count(1)
                if number_of_single_bonds_to_c == 2 and \
                        bo_valences[i] == 2:
                    q += 1
                if number_of_single_bonds_to_c == 3 and \
                        q + 1 < charg:
                    q += 2
    if (bo - ac).sum() == sum(du) and charg == q:
        return True
    else:
        return False


def get_atomic_charge(atom, atomic_valence_electrons, bo_valence):
    if atom == 1:
        char = 1 - bo_valence
    elif atom == 5:
        char = 3 - bo_valence
    elif atom == 15 and bo_valence == 5:
        char = 0
    elif atom == 16 and bo_valence == 6:
        char = 0
    else:
        char = atomic_valence_electrons - 8 + bo_valence
          
    return char


def clean_charges(molec):
    """
    this is a temporary hack. The real solution is to
    generate several BO matrices in ac2bo and pick the one
    with the lowest number of atomic charges
    """
    rxn_smarts = ['[N+:1]=[*:2]-[O-:3]>>[N+0:1]-[*:2]=[O-0:3]',
                  '[N+:1]=[*:2]-[*:3]=[*:4]-[O-:5]>>'
                  '[N+0:1]-[*:2]=[*:3]-[*:4]=[O-0:5]']

    fragments = Chem.GetMolFrags(molec, asMols=True)

    for i, fragment in enumerate(fragments):
        for smarts in rxn_smarts:
            patt = Chem.MolFromSmarts(smarts.split(">>")[0])
            while fragment.HasSubstructMatch(patt):
                rxn = AllChem.ReactionFromSmarts(smarts)
                ps = rxn.RunReactants((fragment,))
                fragment = ps[0][0]
        if i == 0:
            molec = fragment
        else:
            molec = Chem.CombineMols(molec, fragment)
                        
    return molec


def bo2mol(mol, bo_matrix, atomic_num_list,
           atomic_valence_electrons, mol_charge,
           charged_fragments):
    """
    based on code written by Paolo Toscani
    """

    bo_len = len(bo_matrix)
    l2 = len(atomic_num_list)
    bo_valences = list(bo_matrix.sum(axis=1))

    if bo_len != l2:
        raise RuntimeError('sizes of adjMat ({0:d}) '
                           'and atomicNumList '
                           '{1:d} differ'.format(bo_len, l2))
    rw_mol = Chem.RWMol(mol)

    bond_type_dict = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE}

    for i in range(bo_len):
        for j in range(i + 1, bo_len):
            bo = int(round(bo_matrix[i, j]))
            if bo == 0:
                continue
            bt = bond_type_dict.get(bo, Chem.BondType.SINGLE)
            rw_mol.AddBond(i, j, bt)
    mol = rw_mol.GetMol()

    if charged_fragments:
        mol = set_atomic_charges(mol, atomic_num_list,
                                 atomic_valence_electrons,
                                 bo_valences, bo_matrix, mol_charge)
    else:
        mol = set_atomic_radicals(mol, atomic_num_list,
                                  atomic_valence_electrons,
                                  bo_valences)

    return mol


def set_atomic_charges(mol, atomic_num_list,
                       atomic_valence_electrons,
                       bo_valences, bo_matrix, mol_charge):
    q = 0
    for i, atom in enumerate(atomic_num_list):
        a = mol.GetAtomWithIdx(i)
        charg = get_atomic_charge(atom,
                                  atomic_valence_electrons[atom],
                                  bo_valences[i])
        q += charg
        if atom == 6:
            number_of_single_bonds_to_c = \
                list(bo_matrix[i, :]).count(1)
            if number_of_single_bonds_to_c == 2 and \
                    bo_valences[i] == 2:
                    q += 1
                    charg = 0
            if number_of_single_bonds_to_c == 3 and \
                    q + 1 < mol_charge:
                    q += 2
                    charg = 1

        if abs(charg) > 0:
            a.SetFormalCharge(int(charg))
    # rdmolops.SanitizeMol(mol)
    mol = clean_charges(mol)
    return mol


def set_atomic_radicals(mol, atomic_num_list,
                        atomic_valence_electrons,
                        bo_valences):
    """
    The number of radical electrons = absolute atomic charge
    """
    for i, atom in enumerate(atomic_num_list):
        a = mol.GetAtomWithIdx(i)
        charg = get_atomic_charge(atom,
                                  atomic_valence_electrons[atom],
                                  bo_valences[i])

        if abs(charg) > 0:
            a.SetNumRadicalElectrons(abs(charg))

    return mol


def ac2bo(ac, atomic_num_list, charg, charged_fragments):
    atomic_valence = defaultdict(list)
    atomic_valence[1] = [1]
    atomic_valence[6] = [4]
    atomic_valence[7] = [4, 3]
    # atomic_valence[7] = [3]
    atomic_valence[8] = [2, 1]
    atomic_valence[9] = [1]
    atomic_valence[14] = [4]
    atomic_valence[15] = [5, 4, 3]
    atomic_valence[16] = [6, 4, 2]
    atomic_valence[17] = [1]
    atomic_valence[32] = [4]
    atomic_valence[35] = [1]
    atomic_valence[53] = [1]
    
    atomic_valence_electrons = dict()
    atomic_valence_electrons[1] = 1
    atomic_valence_electrons[6] = 4
    atomic_valence_electrons[7] = 5
    atomic_valence_electrons[8] = 6
    atomic_valence_electrons[9] = 7
    atomic_valence_electrons[14] = 4
    atomic_valence_electrons[15] = 5
    atomic_valence_electrons[16] = 6
    atomic_valence_electrons[17] = 7
    atomic_valence_electrons[32] = 4
    atomic_valence_electrons[35] = 7
    atomic_valence_electrons[53] = 7

    # make a list of valences, e.g. for CO: [[4],[2,1]]
    valences_list_of_lists = []
    for atomicNum in atomic_num_list:
        valences_list_of_lists.append(atomic_valence[atomicNum])

    # convert [[4],[2,1]] to [[4,2],[4,1]]
    valences_list = itertools.product(*valences_list_of_lists)

    best_bo = ac.copy()

    # implemenation of algorithm shown in Figure 2
    # UA: unsaturated atoms
    # DU: degree of unsaturation (u matrix in Figure)
    # best_BO: Bcurr in Figure 
    is_best_bo = False
    for valences in valences_list:
        ac_valence = list(ac.sum(axis=1))
        ua, du_from_ac = get_ua(valences, ac_valence)
        if len(ua) == 0 or bo_is_ok(ac, ac, charg, du_from_ac,
                                    atomic_valence_electrons,
                                    atomic_num_list,
                                    charged_fragments):
            best_bo = ac.copy()
            break
        ua_perm = itertools.permutations(ua)
        for a in ua_perm:
            # ind = np.array(a)
            # du_sort = np.array(du_from_ac)[ind].tolist()
            bo = get_bo(ac, a, du_from_ac, valences)
            if bo_is_ok(bo, ac, charg, du_from_ac,
                        atomic_valence_electrons,
                        atomic_num_list,
                        charged_fragments):
                best_bo = bo.copy()
                is_best_bo = True
                break
            elif bo.sum() > best_bo.sum():
                    best_bo = bo.copy()
                    print('best comb not found')
        if is_best_bo:
            break
    # TODO FIXME: it is possible tha one molecule has more that one bond order
    # doe positive charge in some atoms, next step will be generate a list
    # of bond orders and make a average
    return best_bo, atomic_valence_electrons


def ac2mol(mol, ac, atomic_num_list, charg, charged_fragments):
    # convert AC matrix to bond order (BO) matrix
    bo, atomic_valence_electrons = ac2bo(ac, atomic_num_list, charg,
                                         charged_fragments)

    # add BO connectivity and charge info to mol object
    mol = bo2mol(mol, bo, atomic_num_list,
                 atomic_valence_electrons,
                 charg, charged_fragments)

    return mol


def get_proto_mol(atomicnumlist):
    mol = Chem.MolFromSmarts("[#" + str(atomicnumlist[0]) + "]")
    rw_mol = Chem.RWMol(mol)
    for i in range(1, len(atomicnumlist)):
        a = Chem.Atom(atomicnumlist[i])
        rw_mol.AddAtom(a)
    
    mol = rw_mol.GetMol()

    return mol


def get_atomic_numlist(atomic_symbols):
    symbol2number = dict()
    symbol2number["H"] = 1
    symbol2number["C"] = 6
    symbol2number["N"] = 7
    symbol2number["O"] = 8
    symbol2number["F"] = 9
    symbol2number["Si"] = 14
    symbol2number["P"] = 15
    symbol2number["S"] = 16
    symbol2number["Cl"] = 17
    symbol2number["Ge"] = 32
    symbol2number["Br"] = 35
    symbol2number["I"] = 53
    
    atomic_num_list = []
    
    for symbol in atomic_symbols:
        atomic_num_list.append(symbol2number[symbol])
    
    return atomic_num_list


def read_xyz_file(file_name):
    atomic_symbols = []
    xyz_coordinates = []
    charg = 0
    with open(file_name, "r") as f:
        for line_number, line in enumerate(f):
            if line_number == 0:
                num_atoms = int(line)
            elif line_number == 1:
                if "charge=" in line:
                    charg = int(line.split("=")[1])
                else:
                    charg = 0
            else:
                atomic_symbol, x, y, z = line.split()
                atomic_symbols.append(atomic_symbol)
                xyz_coordinates.append([float(x), float(y),
                                        float(z)])
    atomic_num_list = get_atomic_numlist(atomic_symbols)
    return atomic_num_list, charg, xyz_coordinates


def xyz2ac(atomic_num_list, xyz):
    import numpy as np
    mol = get_proto_mol(atomic_num_list)

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (xyz[i][0], xyz[i][1],
                                 xyz[i][2]))
    mol.AddConformer(conf)

    d_mat = Chem.Get3DDistanceMatrix(mol)
    pt = Chem.GetPeriodicTable()

    num_atoms = len(atomic_num_list)
    ac = np.zeros((num_atoms, num_atoms)).astype(int)

    for i in range(num_atoms):
        a_i = mol.GetAtomWithIdx(i)
        rcov_i = pt.GetRcovalent(a_i.GetAtomicNum()) * 1.24
        for j in range(i+1, num_atoms):
            a_j = mol.GetAtomWithIdx(j)
            rcov_j = pt.GetRcovalent(a_j.GetAtomicNum()) * 1.24
            if d_mat[i, j] <= rcov_i + rcov_j:
                ac[i, j] = 1
                ac[j, i] = 1

    return ac, mol


def xyz2mol(atomic_num_list, char, xyz_coordinates,
            charged_fragments):

    # Get atom connectivity (AC) matrix,
    # list of atomic numbers, molecular charge,
    # and mol object with no connectivity information
    ac, mol = xyz2ac(atomic_num_list, xyz_coordinates)

    # Convert AC to bond order matrix and add
    # connectivity and charge info to mol object
    new_mol = ac2mol(mol, ac, atomic_num_list,
                     char, charged_fragments)
    
    return new_mol


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(usage='%(prog)s [options] molecule.xyz')
    parser.add_argument('structure', metavar='structure',
                        type=str)
    args = parser.parse_args()

    filename = args.structure
    char_frag = True
    atomic_numlist, charge, xyz_coord = \
        read_xyz_file(filename)

    molecule = xyz2mol(atomic_numlist, charge, xyz_coord,
                       char_frag)

    # Canonical hack
    smiles = Chem.MolToSmiles(molecule)
    m = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(m)

    print(smiles)
